import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from ml.answer_evaluator import evaluate_answer
from ml.answer_bank import (
    AnswerBankError,
    find_best_match,
    list_questions,
    load_questions_dataset,
)
from ml.doubt_classifier import classify_doubt
from ml.predictor import predict_performance
from ml.qa_engine import answer_question


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

app = Flask(__name__, static_folder=str(FRONTEND_DIST_DIR), static_url_path="")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

try:
    load_questions_dataset()
except AnswerBankError:
    pass


def _request_value(name, default=None):
    json_payload = request.get_json(silent=True)
    if isinstance(json_payload, dict) and name in json_payload:
        return json_payload.get(name, default)
    return request.form.get(name, default)


def _serve_frontend():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(app.static_folder, "index.html")
    return (
        jsonify(
            {
                "success": False,
                "error": (
                    "Frontend build not found. Run `npm install` and `npm run build` "
                    "from the project root, or use `npm run dev` for local development."
                ),
            }
        ),
        503,
    )


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        attendance = float(_request_value("attendance", 0))
        study_hours = float(_request_value("study_hours", 0))
        previous_marks = float(_request_value("previous_marks", 0))
        assignments = int(float(_request_value("assignments_completed", 0)))

        errors = []
        if not (0 <= attendance <= 100):
            errors.append("Attendance must be between 0 and 100.")
        if not (0 <= study_hours <= 24):
            errors.append("Study hours must be between 0 and 24.")
        if not (0 <= previous_marks <= 100):
            errors.append("Previous marks must be between 0 and 100.")
        if not (0 <= assignments <= 10):
            errors.append("Assignments completed must be between 0 and 10.")

        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        result = predict_performance(attendance, study_hours, previous_marks, assignments)
        return jsonify(
            {
                "success": True,
                "predicted_score": result["predicted_score"],
                "risk_level": result["risk"]["level"],
                "risk_color": result["risk"]["color"],
                "risk_emoji": result["risk"]["emoji"],
                "risk_message": result["risk"]["message"],
                "input_summary": result["input_summary"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 500


@app.route("/api/classify-doubt", methods=["POST"])
@app.route("/classify_doubt", methods=["POST"])
def classify_doubt_route():
    try:
        question = (_request_value("question", "") or "").strip()
        print(f"[DEBUG /classify_doubt] Received question: '{question}'")  # DEBUG: Log received question
        if not question:
            return jsonify({"success": False, "error": "Please enter a question."}), 400

        result = classify_doubt(question)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400

        return jsonify(
            {
                "success": True,
                "category": result["category"],
                "confidence": result["confidence"],
                "emoji": result["emoji"],
                "color": result["color"],
                "tip": result["tip"],
                "all_probabilities": result["all_probabilities"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/ask", methods=["POST"])
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = (_request_value("question", "") or "").strip()
        print(f"[DEBUG /ask] Received question: '{question}'")  # DEBUG: Log received question
        if not question:
            return jsonify({"success": False, "error": "Please enter a question."}), 400

        result = answer_question(question)
        print(f"[DEBUG /ask] Answer result matched_question: '{result.get('matched_question', 'N/A')}'")  # DEBUG: Log matched question
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400

        return jsonify({"success": True, **result})
    except Exception as exc:
        print(f"[DEBUG /ask] Exception: {str(exc)}")  # DEBUG: Log exception
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/evaluate", methods=["POST"])
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        question = (_request_value("question", "") or "").strip()
        student_answer = (_request_value("student_answer", "") or "").strip()

        if not question or not student_answer:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please enter a question and provide the student answer.",
                    }
                ),
                400,
            )

        answer_record = find_best_match(question)
        result = evaluate_answer(
            answer_record["question"],
            answer_record["correct_answer"],
            student_answer,
        )
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400

        return jsonify(
            {
                "success": True,
                "question_id": answer_record["question_id"],
                "question": question,
                "matched_question": answer_record["question"],
                "question_match_score": answer_record["match_score"],
                "correct_answer": answer_record["correct_answer"],
                "score": result["score"],
                "semantic_similarity": result["semantic_similarity"],
                "label": result["label"],
                "emoji": result["emoji"],
                "color": result["color"],
                "message": result["message"],
                "word_count": result["word_count"],
                "matched_keywords": result.get("matched_keywords", []),
                "missed_keywords": result.get("missed_keywords", []),
            }
        )
    except AnswerBankError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.get("/api/questions")
@app.get("/questions")
def questions():
    try:
        return jsonify(
            {
                "success": True,
                "questions": list_questions(),
            }
        )
    except AnswerBankError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.get("/")
def home():
    return _serve_frontend()


@app.get("/<path:path>")
def frontend(path):
    asset_path = FRONTEND_DIST_DIR / path
    if asset_path.exists() and asset_path.is_file():
        return send_from_directory(app.static_folder, path)
    return _serve_frontend()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("\nAI Teacher Augmentation System")
    print(f"Frontend/API: http://127.0.0.1:{port}\n")
    app.run(debug=os.getenv("FLASK_DEBUG", "true").lower() == "true", port=port)
