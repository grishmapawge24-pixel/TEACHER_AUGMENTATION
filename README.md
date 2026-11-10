# Teacher Augmentation

A full-stack teaching support app with a React frontend and Flask API. It can
estimate student performance and risk, classify student doubts, answer
subject-related questions, and evaluate answers against a question bank.

## Features

- Predict a student's score and risk level from attendance, study hours,
  previous marks, and completed assignments.
- Classify a question into Python, Web Development, Data Science, or DSA.
- Find a relevant answer from the subject question banks.
- Evaluate a student's answer and show matched and missed keywords.

## Requirements

- Python and pip
- Node.js and npm

## Setup

From the project root, create and activate a Python virtual environment, then
install the backend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the frontend dependencies:

```powershell
npm install
```

The `.env.example` files document optional local configuration. Copy them to
`.env` in the project root and/or `frontend/.env` if you need to change the
defaults. Do not commit files containing local secrets.

## Run locally

Open two terminals in the project root. In the first, start the Flask API:

```powershell
python app.py
```

In the second, start the Vite development server:

```powershell
npm run dev:frontend
```

Open <http://127.0.0.1:5173>. The Vite server proxies `/api` requests to the
backend at <http://127.0.0.1:5000>.

To build the frontend for Flask to serve, run:

```powershell
npm run build
```

Then start Flask and open <http://127.0.0.1:5000>.

## API

The API accepts JSON requests at:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/predict` | Predict performance and risk |
| `POST` | `/api/classify-doubt` | Classify a student's question |
| `POST` | `/api/ask` | Find an answer to a question |
| `POST` | `/api/evaluate` | Evaluate a student's answer |
| `GET` | `/api/questions` | List available evaluation questions |

## Project layout

- `app.py` — Flask application and API routes
- `frontend/` — React, TypeScript, and Vite application
- `ml/` — prediction, classification, question-answering, and evaluation logic
- `models/` — saved machine-learning models
- `data/` — question banks and sample student data
- `templates/` — legacy HTML template

The student dataset contains attendance and grade-related records. Treat it as
sensitive data and avoid publishing personally identifiable student
information.
