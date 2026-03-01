# Dr. Cinema — NY Times Top 50 Movies Recommender

**Author:** Huckjun Hong (hh2772)
**Course:** IEOR 4576 — Agentic AI for Analytics  
**Project:** Project 1 (Domain Q&A Chatbot)

## About
Dr. Cinema is a chatbot that can help you find your next favorite film from the 
New York Times' list of 50 greatest movies in the 21st century. Tell Dr. Cinema your 
mood, a theme, a genre, or a director — and get personalized recommendations instantly.

## Live URL
*(add after deployment)*

## How to Run Locally
Python, uv, and the gcloud CLI are required. A GCP project with Vertex AI enabled and application default credentials need to be set up.

### 1. Clone the repo
```
git clone 
cd movie-chatbot
```

### 2. Log in with GCP credentials
```
gcloud auth application-default login
gcloud auth application-default set-quota-project proejct ID
```

### 3. Set your project ID in main.py
Open app/main.py and update this specific line:
```
GCP_PROJECT = os.getenv("GCP_PROJECT", "project ID")
```

### 4. Install dependencies and run
```
uv sync
uv run uvicorn app.main:app --reload
```

### 5. Open in browser
http://localhost:8000

## How to Run Evaluations
Once the server is first running, open a second terminal:
```
uv run python eval/run_eval.py
```
Results will be saled to eval/results.json

## Project Structure

movie-chatbot/
- pyproject.toml           # dependencies
- movies.json              # NYT Top 50 movies dataset  
- app/
    - main.py              # FastAPI backend
    - prompt.txt           # system prompt
    - templates/
        - index.html       # chat UI
- eval/
    - golden_dataset.json  # 20 test cases
    - run_eval.py          # eval script