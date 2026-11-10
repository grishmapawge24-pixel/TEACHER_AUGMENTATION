# =============================================================
# qa_engine.py
# PURPOSE: A completely independent QA system that:
#   1. Classifies the subject of a student's question
#   2. Routes it to the correct subject-specific QA model
#   3. Finds the best matching answer using TF-IDF + cosine similarity
#
# COMPLETELY ISOLATED from predictor.py and doubt_classifier.py
# No shared state, no shared imports, no shared models.
# =============================================================

import pandas as pd
import joblib
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# SUBJECT CONFIGURATION
# Each subject has: csv file, emoji, color, tip
# To add a new subject: add an entry here + create its CSV
# -------------------------------------------------------
SUBJECTS = {
    "Python": {
        "csv":   "data/qa_python.csv",
        "emoji": "🐍",
        "color": "success",
        "tip":   "Practice by writing small programs daily."
    },
    "Web Development": {
        "csv":   "data/qa_webdev.csv",
        "emoji": "🌐",
        "color": "primary",
        "tip":   "Build small projects to reinforce concepts."
    },
    "Data Science": {
        "csv":   "data/qa_datascience.csv",
        "emoji": "📊",
        "color": "warning",
        "tip":   "Work through real datasets on Kaggle."
    },
    "DSA": {
        "csv":   "data/qa_dsa.csv",
        "emoji": "🧩",
        "color": "danger",
        "tip":   "Solve 1-2 problems on LeetCode daily."
    },
}

# -------------------------------------------------------
# MODELS — loaded once when this file is imported
# -------------------------------------------------------

# Model A: Subject Classifier (tells us WHICH subject)
CLASSIFIER_PATH = "models/qa_subject_classifier.pkl"

# Model B: Per-subject QA vectorizers and answer banks
# Structure: { "Python": {"vectorizer": ..., "questions": [...], "answers": [...]} }
QA_BANKS = {}


def _build_and_save_models():
    """
    Trains and saves:
    1. The subject classifier
    2. Per-subject TF-IDF vectorizers
    Called automatically if models don't exist yet.
    """
    print("[QAEngine] Building models from scratch...")

    # --- Part 1: Build subject classifier ---
    all_questions = []
    all_labels = []

    for subject, config in SUBJECTS.items():
        df = pd.read_csv(config["csv"])
        all_questions.extend(df["question"].tolist())
        all_labels.extend([subject] * len(df))

    classifier_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    classifier_pipeline.fit(all_questions, all_labels)
    joblib.dump(classifier_pipeline, CLASSIFIER_PATH)
    print(f"[QAEngine] ✅ Subject classifier saved → {CLASSIFIER_PATH}")

    # --- Part 2: Build per-subject QA vectorizers ---
    qa_data = {}
    for subject, config in SUBJECTS.items():
        df = pd.read_csv(config["csv"])
        questions = df["question"].tolist()
        answers   = df["answer"].tolist()

        # TF-IDF vectorizer fitted on THIS subject's questions only
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        vectorizer.fit(questions)

        qa_data[subject] = {
            "vectorizer": vectorizer,
            "questions":  questions,
            "answers":    answers,
        }
        print(f"[QAEngine] ✅ QA bank built for {subject} ({len(questions)} Q&A pairs)")

    joblib.dump(qa_data, "models/qa_banks.pkl")
    print("[QAEngine] ✅ All QA banks saved → models/qa_banks.pkl")

    return classifier_pipeline, qa_data


def _load_models():
    """Loads pre-built models from disk, or builds them if missing."""
    global QA_BANKS

    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists("models/qa_banks.pkl"):
        print("[QAEngine] Models not found — building now...")
        classifier, QA_BANKS = _build_and_save_models()
    else:
        print("[QAEngine] Loading existing models from disk...")
        classifier = joblib.load(CLASSIFIER_PATH)
        QA_BANKS   = joblib.load("models/qa_banks.pkl")
        print("[QAEngine] ✅ All models loaded successfully!")

    return classifier


# Load everything when this module is imported
CLASSIFIER = _load_models()


# -------------------------------------------------------
# PUBLIC FUNCTION: answer_question()
# This is the only function Flask needs to call
# -------------------------------------------------------
def answer_question(question: str) -> dict:
    """
    Main entry point. Takes any programming question and returns:
    - The detected subject
    - The best matching answer from the QA bank
    - Confidence score
    - Emoji and color for the UI
    - Top 3 related questions

    Args:
        question (str): Any student question

    Returns:
        dict with all fields needed by Flask/frontend
    """
    if not question or len(question.strip()) < 3:
        return {"error": "Please enter a valid question (at least 3 characters)"}

    question = question.strip()
    print(f"[DEBUG answer_question] Processing question: '{question}'")  # DEBUG: Log question

    # -------------------------------------------------------
    # STEP 1: Classify the subject
    # -------------------------------------------------------
    subject = CLASSIFIER.predict([question])[0]
    subject_confidence = max(CLASSIFIER.predict_proba([question])[0]) * 100

    # Get all subject probabilities for the UI breakdown
    all_subjects = CLASSIFIER.classes_.tolist()
    all_probs    = CLASSIFIER.predict_proba([question])[0].tolist()
    subject_probs = {s: round(p * 100, 1) for s, p in zip(all_subjects, all_probs)}
    
    print(f"[DEBUG answer_question] Classified subject: {subject}")  # DEBUG: Log subject

    # -------------------------------------------------------
    # STEP 2: Route to the correct QA bank
    # -------------------------------------------------------
    qa_bank    = QA_BANKS[subject]
    vectorizer = qa_bank["vectorizer"]
    questions  = qa_bank["questions"]
    answers    = qa_bank["answers"]

    # -------------------------------------------------------
    # STEP 3: Find the best matching question using cosine similarity
    # -------------------------------------------------------

    # Vectorize all stored questions + the incoming question
    all_vectors      = vectorizer.transform(questions)
    question_vector  = vectorizer.transform([question])

    # Compute cosine similarity between the input and all stored questions
    similarities = cosine_similarity(question_vector, all_vectors)[0]

    # Get top 3 most similar questions
    top3_indices   = np.argsort(similarities)[::-1][:3]
    best_idx       = top3_indices[0]
    best_score     = similarities[best_idx]

    print(f"[DEBUG answer_question] Best matched question: '{questions[best_idx]}' with score: {best_score}")  # DEBUG: Log match

    # -------------------------------------------------------
    # STEP 4: Package and return the result
    # -------------------------------------------------------
    subject_config = SUBJECTS[subject]

    # If similarity is very low the question might be off-topic
    if best_score < 0.05:
        answer = (
            f"I couldn't find a specific answer for that in my {subject} knowledge base. "
            f"Try rephrasing your question or ask something more specific about {subject}."
        )
        confidence = round(best_score * 100, 1)
    else:
        answer     = answers[best_idx]
        confidence = round(best_score * 100, 1)

    # Build related questions list (excluding the best match)
    related = [
        {"question": questions[i], "similarity": round(similarities[i] * 100, 1)}
        for i in top3_indices[1:]
        if similarities[i] > 0.01
    ]

    return {
        "subject":          subject,
        "subject_emoji":    subject_config["emoji"],
        "subject_color":    subject_config["color"],
        "subject_tip":      subject_config["tip"],
        "subject_confidence": round(subject_confidence, 1),
        "subject_probs":    subject_probs,

        "answer":           answer,
        "answer_confidence": confidence,
        "matched_question": questions[best_idx],

        "related":          related,
    }