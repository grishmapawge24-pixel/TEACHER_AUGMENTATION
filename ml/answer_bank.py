import csv
import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CANDIDATES = [
    BASE_DIR / "data" / "questions.json",
    BASE_DIR / "data" / "questions.csv",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


class AnswerBankError(Exception):
    pass


def _resolve_dataset_path() -> Path:
    configured_path = os.getenv("QUESTIONS_DATASET_PATH", "").strip()
    if configured_path:
        dataset_path = Path(configured_path)
        if not dataset_path.is_absolute():
            dataset_path = BASE_DIR / dataset_path
        return dataset_path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_DATASET_CANDIDATES[0]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _tokenize(text: str) -> list[str]:
    normalized_text = _normalize_text(text)
    return [token for token in normalized_text.split() if token and token not in STOPWORDS]


def _parse_keywords(raw_keywords) -> list[str]:
    if isinstance(raw_keywords, list):
        values = raw_keywords
    else:
        values = str(raw_keywords or "").replace(",", "|").split("|")

    keywords = []
    for value in values:
        keyword = _normalize_text(str(value))
        if keyword:
            keywords.append(keyword)
    return keywords


def _normalize_record(record: dict) -> dict:
    question_id = str(record.get("id") or record.get("question_id") or "").strip()
    question = str(record.get("question") or "").strip()
    correct_answer = str(record.get("correct_answer") or "").strip()
    keywords = _parse_keywords(record.get("keywords", []))

    if not question_id or not question or not correct_answer:
        raise AnswerBankError(
            "Each dataset record must include id, question, and correct_answer."
        )

    question_tokens = set(_tokenize(question))
    keyword_tokens = set()
    for keyword in keywords:
        keyword_tokens.update(_tokenize(keyword))

    return {
        "id": question_id,
        "question_id": question_id,
        "question": question,
        "correct_answer": correct_answer,
        "keywords": keywords,
        "normalized_question": _normalize_text(question),
        "question_tokens": question_tokens,
        "keyword_tokens": keyword_tokens,
    }


def _load_json_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8-sig") as dataset_file:
        records = json.load(dataset_file)

    if not isinstance(records, list):
        raise AnswerBankError("JSON dataset must contain a list of question records.")

    return [_normalize_record(record) for record in records]


def _load_csv_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames is None:
            raise AnswerBankError("CSV dataset must include a header row.")
        return [_normalize_record(record) for record in reader]


@lru_cache(maxsize=1)
def load_questions_dataset() -> tuple[Path, list[dict]]:
    dataset_path = _resolve_dataset_path()
    if not dataset_path.exists():
        raise AnswerBankError(f"Dataset file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        records = _load_json_dataset(dataset_path)
    elif suffix == ".csv":
        records = _load_csv_dataset(dataset_path)
    else:
        raise AnswerBankError(
            f"Unsupported dataset format '{dataset_path.suffix}'. Use .json or .csv."
        )

    if len(records) == 0:
        raise AnswerBankError("Dataset file is empty.")

    return dataset_path, records


def get_dataset_metadata() -> dict:
    dataset_path, records = load_questions_dataset()
    return {
        "path": str(dataset_path),
        "format": dataset_path.suffix.lower().lstrip("."),
        "question_count": len(records),
    }


def list_questions() -> list[dict]:
    _, records = load_questions_dataset()
    return [{"id": record["id"], "question": record["question"]} for record in records]


def find_best_match(user_question: str, min_score: float = 0.18) -> dict:
    normalized_question = (user_question or "").strip()
    if not normalized_question:
        raise AnswerBankError("Question is required.")

    user_tokens = set(_tokenize(normalized_question))
    normalized_user_question = _normalize_text(normalized_question)

    if not user_tokens and not normalized_user_question:
        raise AnswerBankError("Question is too short to match.")

    _, records = load_questions_dataset()
    best_record = None
    best_score = 0.0

    for record in records:
        token_overlap = (
            len(user_tokens & record["question_tokens"]) / max(len(user_tokens), 1)
            if user_tokens
            else 0.0
        )
        keyword_overlap = (
            len(user_tokens & record["keyword_tokens"]) / max(len(record["keyword_tokens"]), 1)
            if record["keyword_tokens"]
            else 0.0
        )
        sequence_score = SequenceMatcher(
            None, normalized_user_question, record["normalized_question"]
        ).ratio()
        substring_bonus = 1.0 if normalized_user_question in record["normalized_question"] else 0.0

        score = (
            0.45 * token_overlap
            + 0.3 * keyword_overlap
            + 0.2 * sequence_score
            + 0.05 * substring_bonus
        )

        if score > best_score:
            best_score = score
            best_record = record

    if best_record is None or best_score < min_score:
        raise AnswerBankError("No similar question found.")

    return {
        **best_record,
        "match_score": round(best_score * 100, 1),
    }
