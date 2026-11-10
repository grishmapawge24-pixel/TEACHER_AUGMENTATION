import { FormEvent, useState } from "react";

type PredictionResponse = {
  success: boolean;
  predicted_score: number;
  risk_level: string;
  risk_color: "danger" | "warning" | "success";
  risk_emoji: string;
  risk_message: string;
  input_summary: {
    attendance: number;
    study_hours: number;
    previous_marks: number;
    assignments_completed: number;
  };
  errors?: string[];
};

type ClassifyResponse = {
  success: boolean;
  category: string;
  confidence: number;
  emoji: string;
  color: string;
  tip: string;
  all_probabilities: Record<string, number>;
  error?: string;
};

type AskResponse = {
  success: boolean;
  subject: string;
  subject_emoji: string;
  subject_color: string;
  subject_tip: string;
  subject_confidence: number;
  subject_probs: Record<string, number>;
  answer: string;
  answer_confidence: number;
  matched_question: string;
  related: Array<{ question: string; similarity: number }>;
  error?: string;
};

type EvaluateResponse = {
  success: boolean;
  question_id: string;
  question: string;
  matched_question: string;
  question_match_score: number;
  correct_answer: string;
  score: number;
  semantic_similarity: number;
  label: string;
  emoji: string;
  color: "danger" | "primary" | "success";
  message: string;
  word_count: {
    model: number;
    student: number;
  };
  matched_keywords: string[];
  missed_keywords: string[];
  error?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

const predictionDefaults = {
  attendance: 78,
  study_hours: 5,
  previous_marks: 72,
  assignments_completed: 8,
};

async function apiRequest<T>(path: string, payload?: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: payload ? "POST" : "GET",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(
      data.error || (Array.isArray(data.errors) ? data.errors.join(", ") : "Request failed."),
    );
  }
  return data as T;
}

function SectionHeader({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle: string }) {
  return (
    <div className="section-head">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </div>
  );
}

function App() {
  const [predictionForm, setPredictionForm] = useState(predictionDefaults);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);

  const [learningQuestion, setLearningQuestion] = useState("");
  const [learningLoading, setLearningLoading] = useState(false);
  const [learningError, setLearningError] = useState("");
  const [classifyResult, setClassifyResult] = useState<ClassifyResponse | null>(null);
  const [qaResult, setQaResult] = useState<AskResponse | null>(null);

  const [evalForm, setEvalForm] = useState({
    question: "",
    student_answer: "",
  });
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState("");
  const [evalResult, setEvalResult] = useState<EvaluateResponse | null>(null);

  async function handlePredictionSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPredictionLoading(true);
    setPredictionError("");

    try {
      const data = await apiRequest<PredictionResponse>("/predict", predictionForm);
      setPredictionResult(data);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction failed.");
    } finally {
      setPredictionLoading(false);
    }
  }

  async function handleLearningSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLearningLoading(true);
    setLearningError("");

    try {
      const [classification, answer] = await Promise.all([
        apiRequest<ClassifyResponse>("/classify-doubt", {
          question: learningQuestion,
        }),
        apiRequest<AskResponse>("/ask", {
          question: learningQuestion,
        }),
      ]);

      setClassifyResult(classification);
      setQaResult(answer);
    } catch (error) {
      setLearningError(error instanceof Error ? error.message : "Question lookup failed.");
    } finally {
      setLearningLoading(false);
    }
  }

  async function handleEvaluateSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setEvalLoading(true);
    setEvalError("");

    try {
      const data = await apiRequest<EvaluateResponse>("/evaluate", evalForm);
      setEvalResult(data);
    } catch (error) {
      setEvalError(error instanceof Error ? error.message : "Evaluation failed.");
    } finally {
      setEvalLoading(false);
    }
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
          --bg: #eff3fb;
          --surface: rgba(255,255,255,0.82);
          --surface-strong: #ffffff;
          --stroke: rgba(79, 70, 229, 0.12);
          --stroke-strong: rgba(79, 70, 229, 0.2);
          --text: #181c33;
          --muted: #65708f;
          --accent: #4f46e5;
          --accent-soft: #eef0ff;
          --danger: #e11d48;
          --warning: #d97706;
          --success: #059669;
          --shadow: 0 22px 60px rgba(84, 97, 151, 0.14);
          --radius: 24px;
        }

        * { box-sizing: border-box; }

        body {
          margin: 0;
          font-family: 'Space Grotesk', sans-serif;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(79,70,229,0.16), transparent 32%),
            radial-gradient(circle at top right, rgba(13,148,136,0.12), transparent 24%),
            linear-gradient(180deg, #f8faff 0%, #eef3fb 100%);
        }

        button, input, textarea, select { font: inherit; }

        .page {
          min-height: 100vh;
          padding: 32px 20px 64px;
        }

        .shell {
          max-width: 1280px;
          margin: 0 auto;
        }

        .hero {
          margin-bottom: 28px;
        }

        .hero-card, .card {
          background: var(--surface);
          backdrop-filter: blur(18px);
          border: 1px solid var(--stroke);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
        }

        .hero-card {
          padding: 30px;
          position: relative;
          overflow: hidden;
        }

        .hero-card::before {
          content: "";
          position: absolute;
          inset: auto -80px -80px auto;
          width: 220px;
          height: 220px;
          border-radius: 999px;
          background: radial-gradient(circle, rgba(79,70,229,0.22), transparent 70%);
        }

        .chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border-radius: 999px;
          padding: 8px 14px;
          font-size: 13px;
        }

        .hero h1 {
          margin: 0 0 12px;
          font-size: clamp(2.8rem, 6vw, 5rem);
          line-height: 1;
          letter-spacing: -0.05em;
        }

        .hero-card {
          padding: 48px 30px;
          text-align: center;
        }

        .hero h2 {
          margin: 0;
          font-size: clamp(1.1rem, 2.6vw, 1.7rem);
          font-weight: 500;
          color: var(--muted);
        }

        .card h3 {
          margin: 0;
          font-size: 22px;
        }

        .section-head p, .helper {
          margin: 0;
          color: var(--muted);
          line-height: 1.6;
        }

        .section-head {
          margin: 8px 0 18px;
        }

        .eyebrow {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: var(--accent);
          letter-spacing: 0.16em;
          text-transform: uppercase;
          margin-bottom: 8px;
        }

        .section-head h2 {
          margin: 0 0 6px;
          font-size: 30px;
          letter-spacing: -0.04em;
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 22px;
        }

        .card {
          padding: 24px;
        }

        .card-head {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          align-items: flex-start;
          margin-bottom: 18px;
        }

        .muted-code {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: var(--muted);
        }

        .form-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }

        .field {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .field-wide {
          grid-column: 1 / -1;
        }

        label {
          font-size: 13px;
          font-weight: 600;
          color: #39405f;
        }

        input[type="range"] {
          width: 100%;
          accent-color: var(--accent);
        }

        input[type="number"], textarea, select {
          width: 100%;
          border-radius: 16px;
          border: 1px solid var(--stroke);
          background: rgba(255,255,255,0.86);
          color: var(--text);
          padding: 14px 16px;
          outline: none;
          transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input[type="number"]:focus, textarea:focus, select:focus {
          border-color: rgba(79,70,229,0.42);
          box-shadow: 0 0 0 4px rgba(79,70,229,0.08);
        }

        textarea {
          min-height: 108px;
          resize: vertical;
        }

        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .slider-value {
          min-width: 56px;
          text-align: center;
          padding: 8px 12px;
          border-radius: 14px;
          background: var(--accent-soft);
          color: var(--accent);
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
        }

        .button-row {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 18px;
        }

        .btn {
          border: none;
          border-radius: 16px;
          padding: 13px 18px;
          font-weight: 700;
          cursor: pointer;
          transition: transform 0.15s ease, opacity 0.15s ease;
        }

        .btn:hover { transform: translateY(-1px); }
        .btn:disabled { opacity: 0.65; cursor: wait; transform: none; }

        .btn-primary {
          background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
          color: white;
        }

        .btn-secondary {
          background: rgba(79,70,229,0.08);
          color: var(--accent);
          border: 1px solid rgba(79,70,229,0.14);
        }

        .notice {
          margin-top: 16px;
          border-radius: 18px;
          padding: 14px 16px;
          font-size: 14px;
          line-height: 1.6;
        }

        .notice-error {
          background: rgba(225,29,72,0.08);
          color: var(--danger);
          border: 1px solid rgba(225,29,72,0.16);
        }

        .placeholder, .result-panel {
          border-radius: 22px;
          border: 1px solid var(--stroke);
          background: rgba(255,255,255,0.78);
          padding: 22px;
        }

        .placeholder {
          min-height: 220px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          color: var(--muted);
        }

        .score-shell {
          display: flex;
          align-items: center;
          gap: 18px;
          margin-bottom: 18px;
        }

        .score-ring {
          width: 104px;
          height: 104px;
          border-radius: 999px;
          display: grid;
          place-items: center;
          border: 6px solid var(--accent);
          background: rgba(79,70,229,0.08);
          flex-shrink: 0;
        }

        .score-ring.success { border-color: var(--success); background: rgba(5,150,105,0.08); }
        .score-ring.warning { border-color: var(--warning); background: rgba(217,119,6,0.08); }
        .score-ring.danger { border-color: var(--danger); background: rgba(225,29,72,0.08); }
        .score-ring.primary { border-color: var(--accent); background: rgba(79,70,229,0.08); }

        .score-ring strong {
          display: block;
          font-size: 28px;
          line-height: 1;
        }

        .score-ring span {
          display: block;
          margin-top: 4px;
          font-size: 12px;
          color: var(--muted);
        }

        .risk-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          border-radius: 999px;
          font-size: 13px;
          font-weight: 700;
        }

        .risk-badge.success { background: rgba(5,150,105,0.1); color: var(--success); }
        .risk-badge.warning { background: rgba(217,119,6,0.1); color: var(--warning); }
        .risk-badge.danger { background: rgba(225,29,72,0.1); color: var(--danger); }
        .risk-badge.primary { background: rgba(79,70,229,0.1); color: var(--accent); }

        .summary-grid, .mini-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          margin-top: 16px;
        }

        .mini-card {
          border-radius: 18px;
          padding: 14px;
          background: rgba(79,70,229,0.04);
          border: 1px solid rgba(79,70,229,0.08);
        }

        .mini-card strong {
          display: block;
          font-size: 18px;
          margin-top: 4px;
        }

        .bar-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-top: 14px;
        }

        .bar-label {
          display: flex;
          justify-content: space-between;
          font-size: 13px;
          margin-bottom: 6px;
        }

        .track {
          width: 100%;
          height: 8px;
          border-radius: 999px;
          background: #e6ebf7;
          overflow: hidden;
        }

        .fill {
          height: 100%;
          border-radius: inherit;
          background: linear-gradient(90deg, #4f46e5 0%, #14b8a6 100%);
        }

        .chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .chip {
          background: rgba(79,70,229,0.08);
          color: var(--accent);
          border: 1px solid rgba(79,70,229,0.12);
          font-weight: 600;
        }

        .chip.good {
          color: var(--success);
          background: rgba(5,150,105,0.08);
          border-color: rgba(5,150,105,0.14);
        }

        .chip.bad {
          color: var(--danger);
          background: rgba(225,29,72,0.08);
          border-color: rgba(225,29,72,0.14);
        }

        @media (max-width: 1024px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 720px) {
          .page { padding-inline: 14px; }
          .hero-card, .card { padding: 20px; }
          .form-grid, .summary-grid, .mini-grid { grid-template-columns: 1fr; }
          .score-shell { align-items: flex-start; flex-direction: column; }
        }
      `}</style>

      <main className="page">
        <div className="shell">
          <section className="hero">
            <div className="hero-card">
              <h1>GradeGaze</h1>
              <h2>AI insights for student performance</h2>
            </div>
          </section>

          <SectionHeader
            eyebrow="Integrated Modules"
            title="One dashboard, four live workflows"
            subtitle="Each card below sends real requests to Flask and renders backend responses directly in the redesigned UI."
          />

          <section className="dashboard-grid">
            <div className="card">
              <div className="card-head">
                <div>
                  <h3>Performance Prediction</h3>
                </div>
              </div>

              <form onSubmit={handlePredictionSubmit}>
                <div className="form-grid">
                  <div className="field">
                    <label htmlFor="attendance">Attendance (%)</label>
                    <div className="slider-row">
                      <input
                        id="attendance"
                        type="range"
                        min="0"
                        max="100"
                        value={predictionForm.attendance}
                        onChange={(event) =>
                          setPredictionForm((current) => ({
                            ...current,
                            attendance: Number(event.target.value),
                          }))
                        }
                      />
                      <div className="slider-value">{predictionForm.attendance}</div>
                    </div>
                  </div>

                  <div className="field">
                    <label htmlFor="study-hours">Study Hours / Day</label>
                    <div className="slider-row">
                      <input
                        id="study-hours"
                        type="range"
                        min="0"
                        max="12"
                        step="0.5"
                        value={predictionForm.study_hours}
                        onChange={(event) =>
                          setPredictionForm((current) => ({
                            ...current,
                            study_hours: Number(event.target.value),
                          }))
                        }
                      />
                      <div className="slider-value">{predictionForm.study_hours}</div>
                    </div>
                  </div>

                  <div className="field">
                    <label htmlFor="marks">Previous Marks</label>
                    <input
                      id="marks"
                      type="number"
                      min="0"
                      max="100"
                      value={predictionForm.previous_marks}
                      onChange={(event) =>
                        setPredictionForm((current) => ({
                          ...current,
                          previous_marks: Number(event.target.value),
                        }))
                      }
                    />
                  </div>

                  <div className="field">
                    <label htmlFor="assignments">Assignments Completed</label>
                    <div className="slider-row">
                      <input
                        id="assignments"
                        type="range"
                        min="0"
                        max="10"
                        value={predictionForm.assignments_completed}
                        onChange={(event) =>
                          setPredictionForm((current) => ({
                            ...current,
                            assignments_completed: Number(event.target.value),
                          }))
                        }
                      />
                      <div className="slider-value">{predictionForm.assignments_completed}</div>
                    </div>
                  </div>
                </div>

                <div className="button-row">
                  <button className="btn btn-primary" type="submit" disabled={predictionLoading}>
                    {predictionLoading ? "Running prediction..." : "Run AI Prediction"}
                  </button>
                  <button
                    className="btn btn-secondary"
                    type="button"
                    onClick={() => {
                      setPredictionForm(predictionDefaults);
                      setPredictionError("");
                    }}
                  >
                    Reset
                  </button>
                </div>
              </form>

              {predictionError ? <div className="notice notice-error">{predictionError}</div> : null}

              {predictionResult ? (
                <div className="result-panel">
                  <div className="score-shell">
                    <div className={`score-ring ${predictionResult.risk_color}`}>
                      <div>
                        <strong>{predictionResult.predicted_score}</strong>
                        <span>/ 100</span>
                      </div>
                    </div>
                    <div>
                      <div className={`risk-badge ${predictionResult.risk_color}`}>
                        <span>{predictionResult.risk_emoji}</span>
                        {predictionResult.risk_level}
                      </div>
                      <p className="helper" style={{ marginTop: 12 }}>
                        {predictionResult.risk_message}
                      </p>
                    </div>
                  </div>

                  <div className="bar-list">
                    {[
                      ["Attendance", predictionResult.input_summary.attendance, "%"],
                      ["Study hours", (predictionResult.input_summary.study_hours / 12) * 100, `${predictionResult.input_summary.study_hours}h`],
                      ["Previous marks", predictionResult.input_summary.previous_marks, "/100"],
                      [
                        "Assignments",
                        (predictionResult.input_summary.assignments_completed / 10) * 100,
                        `${predictionResult.input_summary.assignments_completed}/10`,
                      ],
                    ].map(([label, value, suffix]) => (
                      <div key={String(label)}>
                        <div className="bar-label">
                          <span>{label}</span>
                          <span>{suffix}</span>
                        </div>
                        <div className="track">
                          <div className="fill" style={{ width: `${value}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="placeholder">Submit student data to see the prediction and risk profile.</div>
              )}
            </div>

            <div className="card">
              <div className="card-head">
                <div>
                  <h3>Doubt Solving &amp; Subject-Aware Q&amp;A</h3>
                </div>
              </div>

              <form onSubmit={handleLearningSubmit}>
                <div className="field field-wide">
                  <label htmlFor="learning-question">Enter your question</label>
                  <textarea
                    id="learning-question"
                    value={learningQuestion}
                    onChange={(event) => setLearningQuestion(event.target.value)}
                  />
                </div>
                <div className="button-row">
                  <button className="btn btn-primary" type="submit" disabled={learningLoading}>
                    {learningLoading ? "Solving..." : "Solve Question"}
                  </button>
                </div>
              </form>

              {learningError ? <div className="notice notice-error">{learningError}</div> : null}

              {classifyResult && qaResult ? (
                <div className="result-panel">
                  <div className="score-shell">
                    <div className="score-ring primary">
                      <div>
                        <strong>{qaResult.answer_confidence}%</strong>
                        <span>answer match</span>
                      </div>
                    </div>
                    <div>
                      <div className="risk-badge primary">
                        <span>{qaResult.subject_emoji}</span>
                        {qaResult.subject}
                      </div>
                      <p className="helper" style={{ marginTop: 12 }}>
                        {qaResult.subject_tip}
                      </p>
                    </div>
                  </div>

                  <div className="summary-grid">
                    <div className="mini-card">
                      <span className="muted-code">Classified category</span>
                      <strong>{classifyResult.emoji} {classifyResult.category}</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Classification confidence</span>
                      <strong>{classifyResult.confidence}%</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Matched question</span>
                      <strong>{qaResult.matched_question}</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Subject confidence</span>
                      <strong>{qaResult.subject_confidence}%</strong>
                    </div>
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Study tip</p>
                  <div className="notice" style={{ background: "rgba(79,70,229,0.04)", color: "var(--text)", border: "1px solid rgba(79,70,229,0.08)" }}>
                    {classifyResult.tip}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Question breakdown</p>
                  <div className="bar-list">
                    {Object.entries(classifyResult.all_probabilities)
                      .sort(([, left], [, right]) => right - left)
                      .map(([label, value]) => (
                        <div key={label}>
                          <div className="bar-label">
                            <span>{label}</span>
                            <span>{value}%</span>
                          </div>
                          <div className="track">
                            <div className="fill" style={{ width: `${value}%` }} />
                          </div>
                        </div>
                      ))}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Answer</p>
                  <div className="notice" style={{ background: "rgba(79,70,229,0.04)", color: "var(--text)", border: "1px solid rgba(79,70,229,0.08)" }}>
                    {qaResult.answer}
                  </div>

                  {qaResult.related.length ? (
                    <>
                      <p className="muted-code" style={{ marginTop: 14 }}>Related questions</p>
                      <div className="chip-row" style={{ marginTop: 10 }}>
                        {qaResult.related.map((item) => (
                          <button
                            key={item.question}
                            className="chip"
                            type="button"
                            onClick={() => setLearningQuestion(item.question)}
                          >
                            {item.question}
                          </button>
                        ))}
                      </div>
                    </>
                  ) : null}
                </div>
              ) : (
                <div className="placeholder">Ask anything about Python, DSA, Web Development, or Data Science to get classification and an answer in one place.</div>
              )}
            </div>

            <div className="card">
              <div className="card-head">
                <div>
                  <h3>Answer Evaluator</h3>
                </div>
              </div>

              <form onSubmit={handleEvaluateSubmit}>
                <div className="form-grid">
                  <div className="field field-wide">
                    <label htmlFor="eval-question">Enter your question</label>
                    <textarea
                      id="eval-question"
                      value={evalForm.question}
                      onChange={(event) =>
                        setEvalForm((current) => ({ ...current, question: event.target.value }))
                      }
                      style={{ minHeight: 88 }}
                    />
                  </div>
                  <div className="field field-wide">
                    <label htmlFor="eval-student">Student answer</label>
                    <textarea
                      id="eval-student"
                      value={evalForm.student_answer}
                      onChange={(event) =>
                        setEvalForm((current) => ({ ...current, student_answer: event.target.value }))
                      }
                    />
                  </div>
                </div>
                <div className="button-row">
                  <button className="btn btn-primary" type="submit" disabled={evalLoading}>
                    {evalLoading ? "Evaluating..." : "Evaluate Answer"}
                  </button>
                  <button
                    className="btn btn-secondary"
                    type="button"
                    onClick={() => {
                      setEvalForm({
                        question: "",
                        student_answer: "",
                      });
                      setEvalError("");
                    }}
                  >
                    Clear
                  </button>
                </div>
              </form>

              {evalError ? <div className="notice notice-error">{evalError}</div> : null}

              {evalResult ? (
                <div className="result-panel">
                  <div className="score-shell">
                    <div className={`score-ring ${evalResult.color}`}>
                      <div>
                        <strong>{evalResult.score}%</strong>
                        <span>coverage</span>
                      </div>
                    </div>
                    <div>
                      <div className={`risk-badge ${evalResult.color}`}>
                        <span>{evalResult.emoji}</span>
                        {evalResult.label}
                      </div>
                      <p className="helper" style={{ marginTop: 12 }}>
                        {evalResult.message}
                      </p>
                    </div>
                  </div>

                  <div className="summary-grid">
                    <div className="mini-card">
                      <span className="muted-code">Matched question ID</span>
                      <strong>{evalResult.question_id}</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Model words</span>
                      <strong>{evalResult.word_count.model}</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Student words</span>
                      <strong>{evalResult.word_count.student}</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Semantic match</span>
                      <strong>{evalResult.semantic_similarity}%</strong>
                    </div>
                    <div className="mini-card">
                      <span className="muted-code">Question match</span>
                      <strong>{evalResult.question_match_score}%</strong>
                    </div>
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Your typed question</p>
                  <div className="notice" style={{ background: "rgba(79,70,229,0.04)", color: "var(--text)", border: "1px solid rgba(79,70,229,0.08)" }}>
                    {evalResult.question}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Best matched dataset question</p>
                  <div className="notice" style={{ background: "rgba(13,148,136,0.05)", color: "var(--text)", border: "1px solid rgba(13,148,136,0.12)" }}>
                    {evalResult.matched_question}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Stored correct answer</p>
                  <div className="notice" style={{ background: "rgba(5,150,105,0.05)", color: "var(--text)", border: "1px solid rgba(5,150,105,0.12)" }}>
                    {evalResult.correct_answer}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Concepts covered</p>
                  <div className="chip-row" style={{ marginTop: 10 }}>
                    {evalResult.matched_keywords.length ? (
                      evalResult.matched_keywords.map((keyword) => (
                        <span className="chip good" key={keyword}>
                          {keyword}
                        </span>
                      ))
                    ) : (
                      <span className="helper">No strong concept matches found.</span>
                    )}
                  </div>

                  <p className="muted-code" style={{ marginTop: 16 }}>Concepts missed</p>
                  <div className="chip-row" style={{ marginTop: 10 }}>
                    {evalResult.missed_keywords.length ? (
                      evalResult.missed_keywords.map((keyword) => (
                        <span className="chip bad" key={keyword}>
                          {keyword}
                        </span>
                      ))
                    ) : (
                      <span className="helper">All key concepts were covered.</span>
                    )}
                  </div>
                </div>
              ) : (
                <div className="placeholder">Type any question, paste the student answer, and let the backend find the closest stored question automatically.</div>
              )}
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

export default App;
