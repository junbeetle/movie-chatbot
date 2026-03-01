
import json
import os
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import types

# Configuration set ups
CHATBOT_URL = "http://localhost:8000/chat"
GCP_PROJECT = os.getenv("GCP_PROJECT", "hong-agentic-ai-p1")
GCP_LOCATION = "us-central1"
DATASET_PATH = Path("eval/golden_dataset.json")

# Setting up Gemini AI client for judging purposes.
judge_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Loading test cases from the golden dataset JSOn file.
with open(DATASET_PATH) as f:
    TEST_CASES = json.load(f)

# Sending a question to the chatbot and receiving a response.
def call_chatbot(question: str) -> str:
    try:
        resp = httpx.post(
            CHATBOT_URL,
            json={"message": question, "history": []},
            timeout=30,
        )
        data = resp.json()
        return data.get("reply") or data.get("error") or "NO RESPONSE"
    except Exception as e:
        return f"CONNECTION ERROR: {e}"


# Metric 1: Deterministic keyword checks
def deterministic_check(response: str, expected_keywords: list[str]) -> bool:
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in expected_keywords)


# Metric 2a: MaaJ grading with a rubric against expected behavior
def maaj_grade(question: str, response: str, expected_behavior: str) -> tuple[bool, str]:
    judge_prompt = f"""You are grading a movie recommendation chatbot called Dr. Cinema.

Question asked: {question}

Chatbot response: {response}

Expected behavior: {expected_behavior}

Grade whether the chatbot response meets the expected behavior.
Reply with exactly this format:
PASS or FAIL
REASON: one sentence explanation
"""
    
    try:
        result = judge_client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Content(role="user", parts=[types.Part(text=judge_prompt)])],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=100),
        )
        text = result.text.strip()
        passed = text.upper().startswith("PASS")
        reason = text.split("REASON:")[-1].strip() if "REASON:" in text else text
        return passed, reason
    except Exception as e:
        return False, f"Judge error: {e}"


# Metric 2b: Golden reference MaaJ, comparing response to expected keywords
def golden_maaj(question: str, response: str, expected_keywords: list[str]) -> tuple[bool, str]:
    expected_str = ", ".join(expected_keywords)
    judge_prompt = f"""You are grading a movie recommendation chatbot.

Question: {question}

Chatbot response: {response}

Expected answer should reference: {expected_str}

Does the chatbot response appropriately address the question and reference the expected content?
Reply with exactly:
PASS or FAIL
REASON: one sentence
"""
    try:
        result = judge_client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Content(role="user", parts=[types.Part(text=judge_prompt)])],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=100),
        )
        text = result.text.strip()
        passed = text.upper().startswith("PASS")
        reason = text.split("REASON:")[-1].strip() if "REASON:" in text else text
        return passed, reason
    except Exception as e:
        return False, f"Judge error: {e}"


# Running all 20 tests and summary in the end
def run_evals():
    results = []
    category_totals = {}

    print("Dr. Cinema EVALUATION SUMMARY")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    for i, test in enumerate(TEST_CASES, 1):
        test_id  = test["id"]
        category = test["category"]
        question = test["question"]
        keywords = test["expected_keywords"]
        behavior = test["expected_behavior"]

        print(f"[{i:02d}/{len(TEST_CASES)}] {test_id} — {question[:55]}...")

        response = call_chatbot(question)
        time.sleep(1)

        det_pass = deterministic_check(response, keywords)

        rubric_pass, rubric_reason = maaj_grade(question, response, behavior)
        time.sleep(1)

        golden_pass, golden_reason = golden_maaj(question, response, keywords)
        time.sleep(1)
       
        # Considered a pass when 2+ metrics pass.
        metrics_passed = sum([det_pass, golden_pass, rubric_pass])
        overall = metrics_passed >= 2

        status = "PASS" if overall else "FAIL"
        print(f"Deterministic: {'PASS' if det_pass else 'FAIL'} | "
              f"Golden MaaJ: {'PASS' if golden_pass else 'FAIL'} | "
              f"Rubric MaaJ: {'PASS' if rubric_pass else 'FAIL'} => {status}")
        if not overall:
            print(f"Response: {response[:100]}...")
            print(f"Judge note: {rubric_reason}")

        # Tracked by category
        if category not in category_totals:
            category_totals[category] = {"pass": 0, "total": 0}
        category_totals[category]["total"] += 1
        if overall:
            category_totals[category]["pass"] += 1

        results.append({
            "id": test_id,
            "category": category,
            "question": question,
            "response": response,
            "deterministic_pass": det_pass,
            "rubric_maaj_pass": rubric_pass,
            "golden_maaj_pass": golden_pass,
            "overall_pass": overall,
        })

    # Final summary
    total_pass = sum(1 for r in results if r["overall_pass"])
    total = len(results)

    print("RESULTS SUMMARY")

    category_labels = {
        "in_domain": "IN-DOMAIN",
        "out_of_scope": "OUT-OF-SCOPE",
        "adversarial": "ADVERSARIAL",
    }
    for cat, label in category_labels.items():
        if cat in category_totals:
            p = category_totals[cat]["pass"]
            t = category_totals[cat]["total"]
            pct = int(p / t * 100)
            bar = "#" * p + "-" * (t - p)
            print(f"  {label}  [{bar}]  {p}/{t}  ({pct}%)")

    overall_pct = int(total_pass / total * 100)
    print(f"OVERALL {total_pass}/{total} ({overall_pct}%)")

    # Saving results to a file
    out_path = Path("eval/results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {out_path}\n")


if __name__ == "__main__":
    run_evals()
