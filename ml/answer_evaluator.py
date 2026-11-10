# =============================================================
# answer_evaluator.py
# PURPOSE: Compare a student's answer against a model answer
#          and return a meaningful score + feedback label.
#
# TECHNIQUE: Keyword Coverage Scoring with Stemming
#
# HOW IT WORKS:
#   1. Strip all filler/stop words from the model answer
#   2. Stem all remaining words to their root form
#      (sorted → sort, dividing → divid, calls → call)
#   3. Extract topic-anchor words from the QUESTION itself
#   4. Check which of the model's keywords appear in the
#      student's answer (also stemmed, so synonymous forms match)
#   5. Score = weighted blend of:
#       - Keyword coverage  (did they mention the key concepts?)
#       - Question anchors  (did they address the actual topic?)
#
# WHY THIS IS BETTER THAN TF-IDF COSINE:
#   Old approach: "binary search splits a sorted list in half"
#   vs "binary search is an algorithm dividing a sorted array"
#   → TF-IDF cosine scores this LOW because exact tokens differ
#   → This approach scores it HIGH because key concepts match:
#     (search ✓, sort ✓, split/divid ✓, half/halv ✓)
#
# COMPLETELY INDEPENDENT from predictor.py, doubt_classifier.py,
# and qa_engine.py — no shared state, no imports from other modules.
# =============================================================

import re
import math
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# STOPWORDS — expanded list of filler words to ignore
# These carry no conceptual meaning and should not affect scoring
# -------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "us", "him", "her", "them", "my", "our",
    "your", "his", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "any", "all", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "also", "as", "if", "then", "than",
    "into", "up", "out", "about", "after", "before", "while", "during",
    "through", "between", "same", "like", "used", "use", "using",
    "called", "known", "means", "mean", "refer", "refers",
}


# -------------------------------------------------------
# PORTER STEMMER (lightweight, no external dependencies)
#
# A stemmer strips suffixes to reduce words to their root.
# Examples:
#   "sorting"   → "sort"
#   "dividing"  → "divid"
#   "sorted"    → "sort"
#   "functions" → "function"
#   "called"    → "call"
#
# This lets us match "sorts" with "sorted", "calls" with "calling",
# "recursively" with "recursive", etc. — surface form differences
# no longer penalize the student.
# -------------------------------------------------------
def _stem(word: str) -> str:
    """
    Applies a simplified Porter-style stemming to a single word.
    Returns the stemmed root (lowercase).

    This is intentionally lightweight — we want concept matching,
    not perfect morphological analysis. A few over-stems are fine.
    """
    w = word.lower()

    # Remove possessives
    if w.endswith("'s"):
        w = w[:-2]

    # Step 1a: plurals and -ed/-ing
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "i"
    elif w.endswith("ss"):
        pass  # leave as-is
    elif w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
        w = w[:-1]

    # Step 1b: -ed / -ing
    if w.endswith("eed") and len(w) > 4:
        w = w[:-1]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if w.endswith(("at", "bl", "iz")):
            w = w + "e"
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if w.endswith(("at", "bl", "iz")):
            w = w + "e"

    # Step 2: common suffixes
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("isation", "ize"),
        ("ization", "ize"), ("alism", "al"), ("aliti", "al"),
        ("alli", "al"), ("fulness", "ful"), ("ousli", "ous"),
        ("ousness", "ous"), ("iveness", "ive"), ("iviti", "ive"),
        ("biliti", "ble"), ("ically", "ic"), ("icity", "ic"),
        ("ical", "ic"), ("ness", ""), ("ment", ""), ("ly", ""),
        ("er", ""), ("or", ""), ("ion", ""),
    ]
    for suffix, replacement in suffixes:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)] + replacement
            break

    return w


# -------------------------------------------------------
# TOKENIZER — clean text into meaningful word tokens
# -------------------------------------------------------
def _tokenize(text: str) -> list:
    """
    Lowercases, strips punctuation, splits into words,
    removes stopwords and very short tokens (≤2 chars).

    Returns a list of raw (unstemmed) word strings.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation
    tokens = text.split()
    return [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]


def _stemmed_set(text: str) -> set:
    """
    Full pipeline: tokenize → stem → return as a set of root forms.
    Used for concept-level matching (order doesn't matter).
    """
    return {_stem(t) for t in _tokenize(text)}


def _stemmed_list(text: str) -> list:
    """
    Full pipeline: tokenize → stem → return as a list.
    Used for frequency counting (duplicates preserved).
    """
    return [_stem(t) for t in _tokenize(text)]


# -------------------------------------------------------
# KEYWORD IMPORTANCE WEIGHTING
#
# Not all keywords in the model answer are equally important.
# We weight them by their rarity within the answer itself:
# words that appear only once in a long answer are more
# "specific" and worth more than high-frequency words.
#
# We use a simple inverse-document-frequency-style weight:
#   weight(word) = 1 + log(total_tokens / freq_of_word)
# -------------------------------------------------------
def _compute_keyword_weights(stemmed_tokens: list) -> dict:
    """
    Given a list of stemmed tokens from the model answer,
    returns a dict of { stem: weight } where weight reflects
    how informationally important the term is.

    Args:
        stemmed_tokens (list): Output of _stemmed_list(model_answer)

    Returns:
        dict: { "sort": 1.69, "recurs": 2.1, ... }
    """
    if not stemmed_tokens:
        return {}

    freq = Counter(stemmed_tokens)
    total = len(stemmed_tokens)
    weights = {}
    for word, count in freq.items():
        # IDF-inspired: rarer words in this answer → higher weight
        weights[word] = 1 + math.log(total / count)
    return weights


# -------------------------------------------------------
# CONFIGURATION — Scoring thresholds and feedback labels
# -------------------------------------------------------
SCORE_CONFIG = {
    "excellent": {
        "min":     75,
        "label":   "Excellent",
        "emoji":   "🌟",
        "color":   "success",
        "message": "Outstanding! The student has addressed all key concepts from the model answer."
    },
    "good": {
        "min":     50,
        "label":   "Good",
        "emoji":   "✅",
        "color":   "primary",
        "message": "Good effort! The student covers the main idea but is missing some key concepts."
    },
    "needs_improvement": {
        "min":     0,
        "label":   "Needs Improvement",
        "emoji":   "📝",
        "color":   "danger",
        "message": "The answer is missing several key concepts from the model answer. Review the topic carefully."
    },
}

# Blend weights — must sum to 1.0
# keyword_coverage: how many of the model's key terms appear in student answer
# question_anchor:  how many topic words from the question appear in student answer
WEIGHT_KEYWORD_COVERAGE = 0.6
WEIGHT_QUESTION_ANCHOR  = 0.2
WEIGHT_SEMANTIC_SIMILARITY = 0.2


# -------------------------------------------------------
# INTERNAL HELPER: Label from score
# -------------------------------------------------------
def _get_label(score_percent: float) -> dict:
    if score_percent >= SCORE_CONFIG["excellent"]["min"]:
        return SCORE_CONFIG["excellent"]
    elif score_percent >= SCORE_CONFIG["good"]["min"]:
        return SCORE_CONFIG["good"]
    else:
        return SCORE_CONFIG["needs_improvement"]


def _compute_semantic_similarity(model_answer: str, student_answer: str) -> float:
    """
    Computes a lightweight semantic similarity score using TF-IDF vectors.

    This improves evaluation for paraphrased answers that express the same
    idea with different wording.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectors = vectorizer.fit_transform([model_answer, student_answer])
    return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])


# -------------------------------------------------------
# PUBLIC FUNCTION — Flask calls this
# -------------------------------------------------------
def evaluate_answer(question: str, model_answer: str, student_answer: str) -> dict:
    """
    Evaluates a student's answer against the model answer using
    keyword coverage scoring with stemming.

    Scoring logic:
        1. Extract weighted keywords from the model answer
           (stopwords removed, remaining words stemmed)
        2. Extract topic-anchor terms from the question
        3. Check which model keywords appear in the student's
           stemmed answer → weighted coverage score
        4. Check which question anchors appear in the student's
           stemmed answer → anchor coverage score
        5. Final score = weighted blend of keyword coverage, question anchors,
           and semantic similarity between the two answers

    Args:
        question       (str): The original question (topic context)
        model_answer   (str): The correct/expected answer
        student_answer (str): What the student wrote

    Returns:
        dict with keys:
            score           → float 0–100 (keyword coverage %)
            label           → "Excellent" / "Good" / "Needs Improvement"
            emoji           → display emoji
            color           → Bootstrap color class
            message         → Feedback for teacher/student
            matched_keywords→ list of key concepts the student got right
            missed_keywords → list of key concepts the student missed
            word_count      → dict with model and student word counts
    """

    # --- Input validation ---
    question       = question.strip()       if question       else ""
    model_answer   = model_answer.strip()   if model_answer   else ""
    student_answer = student_answer.strip() if student_answer else ""

    if not model_answer:
        return {"error": "Model answer cannot be empty."}
    if not student_answer:
        return {"error": "Student answer cannot be empty."}
    if len(model_answer.split()) < 2:
        return {"error": "Model answer is too short. Please provide at least a sentence."}
    if len(student_answer.split()) < 2:
        return {"error": "Student answer is too short. Please provide at least a sentence."}

    # -------------------------------------------------------
    # STEP 1: Extract and weight keywords from the model answer
    # -------------------------------------------------------
    model_tokens  = _stemmed_list(model_answer)
    keyword_weights = _compute_keyword_weights(model_tokens)
    # keyword_weights = { "sort": 1.4, "recurs": 1.9, "base": 1.6, ... }

    if not keyword_weights:
        return {"error": "Model answer has no meaningful keywords after filtering."}

    # -------------------------------------------------------
    # STEP 2: Build stemmed set of the STUDENT's answer
    # We use a set for O(1) lookup — order doesn't matter here
    # -------------------------------------------------------
    student_stems = _stemmed_set(student_answer)

    # -------------------------------------------------------
    # STEP 3: Keyword Coverage Score
    #
    # For each keyword in the model answer:
    #   - If the student's answer contains its stem → add its weight
    # Coverage = (sum of matched weights) / (sum of all weights)
    #
    # This is smarter than simple word overlap because:
    #   - Rare/specific words (e.g. "halv", "recurs") contribute more
    #   - Common-but-meaningful words contribute less
    #   - Surface form doesn't matter ("sorted" matches "sort")
    # -------------------------------------------------------
    total_weight   = sum(keyword_weights.values())
    matched_weight = 0.0
    matched_keywords = []
    missed_keywords  = []

    for stem, weight in keyword_weights.items():
        if stem in student_stems:
            matched_weight += weight
            matched_keywords.append(stem)
        else:
            missed_keywords.append(stem)

    keyword_coverage_score = matched_weight / total_weight  # 0.0 – 1.0

    # -------------------------------------------------------
    # STEP 4: Question Anchor Score
    #
    # The question tells us what the topic IS.
    # E.g. for "What is binary search?", the anchors are:
    #   "binary" and "search" (after stripping stopwords).
    #
    # If the student mentions these topic words, they're on topic.
    # If they don't appear at all, that's a red flag.
    # -------------------------------------------------------
    question_anchors = _stemmed_set(question)  # e.g. {"binari", "search"}

    if question_anchors:
        anchor_hits = len(question_anchors & student_stems)
        anchor_score = anchor_hits / len(question_anchors)
    else:
        # No question provided → don't penalize
        anchor_score = keyword_coverage_score

    # -------------------------------------------------------
    # STEP 5: Semantic Similarity Score
    #
    # This captures answer closeness even when the student uses
    # different surface wording than the model answer.
    # -------------------------------------------------------
    semantic_similarity = _compute_semantic_similarity(model_answer, student_answer)

    # -------------------------------------------------------
    # STEP 6: Blend into final score
    # -------------------------------------------------------
    raw_score     = (
        WEIGHT_KEYWORD_COVERAGE * keyword_coverage_score
        + WEIGHT_QUESTION_ANCHOR * anchor_score
        + WEIGHT_SEMANTIC_SIMILARITY * semantic_similarity
    )
    score_percent = round(min(raw_score * 100, 100.0), 1)

    # -------------------------------------------------------
    # STEP 7: Build human-readable matched/missed keyword lists
    # De-stem them back to the closest original form for display
    # -------------------------------------------------------
    # Map stems back to original tokens for readable output
    all_model_tokens = _tokenize(model_answer)
    stem_to_original = {}
    for token in all_model_tokens:
        s = _stem(token)
        if s not in stem_to_original:
            stem_to_original[s] = token

    readable_matched = [stem_to_original.get(s, s) for s in matched_keywords]
    readable_missed  = [stem_to_original.get(s, s) for s in missed_keywords]

    # -------------------------------------------------------
    # STEP 8: Return result
    # -------------------------------------------------------
    label_config = _get_label(score_percent)

    return {
        "score":            score_percent,
        "semantic_similarity": round(semantic_similarity * 100, 1),
        "label":            label_config["label"],
        "emoji":            label_config["emoji"],
        "color":            label_config["color"],
        "message":          label_config["message"],
        "matched_keywords": sorted(readable_matched),
        "missed_keywords":  sorted(readable_missed),
        "word_count": {
            "model":   len(model_answer.split()),
            "student": len(student_answer.split()),
        },
    }


# -------------------------------------------------------
# QUICK TEST — Run this file directly:
#   python answer_evaluator.py
# -------------------------------------------------------
if __name__ == "__main__":

    test_cases = [
        {
            "label": "EXCELLENT — same concepts, completely different wording",
            "question": "What is binary search?",
            "model":    "Binary search is an algorithm that finds a target value in a sorted array by repeatedly dividing the search space in half.",
            "student":  "Binary search works by splitting a sorted list in half repeatedly until it locates the target element.",
        },
        {
            "label": "GOOD — covers main idea, misses some concepts",
            "question": "What is a for loop?",
            "model":    "A for loop repeats a block of code a fixed number of times, iterating over a sequence like a list or range.",
            "student":  "A for loop repeats code multiple times over a list.",
        },
        {
            "label": "NEEDS IMPROVEMENT — off-topic answer",
            "question": "What is recursion?",
            "model":    "Recursion is when a function calls itself to solve a smaller version of the same problem, with a base case to stop the recursion.",
            "student":  "It is a type of loop that runs many times.",
        },
        {
            "label": "STEMMING TEST — different word forms, same meaning",
            "question": "How does sorting work?",
            "model":    "Sorting arranges elements in order by comparing values and swapping them into the correct position.",
            "student":  "Elements are sorted by comparing each value and swapped into their correct positions.",
        },
    ]

    print("\n" + "=" * 65)
    print("  ANSWER EVALUATOR — KEYWORD COVERAGE SCORING TEST")
    print("=" * 65)

    for tc in test_cases:
        result = evaluate_answer(tc["question"], tc["model"], tc["student"])
        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}")
            continue
        print(f"\n📌 {tc['label']}")
        print(f"   Score:    {result['score']}%  →  {result['emoji']} {result['label']}")
        print(f"   Message:  {result['message']}")
        print(f"   Matched:  {result['matched_keywords']}")
        print(f"   Missed:   {result['missed_keywords']}")
        print(f"   Words:    Model={result['word_count']['model']}, Student={result['word_count']['student']}")
