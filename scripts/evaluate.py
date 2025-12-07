import csv
from pathlib import Path

import pandas as pd
import requests
import textstat

API_URL = "http://34.26.106.102:8000/answer"  # replace with your VM IP

RAW_CSV = Path("data/raw/medquad_raw.csv")
OUT_CSV = Path("data/eval_with_baselines.csv")

TEST_QUESTIONS = [
    "What causes Glaucoma ?",
    "How to prevent High Blood Pressure ?",
    "What are the symptoms of High Blood Pressure ?",
    "What are the symptoms of Urinary Tract Infections ?",
    "What are the treatments for Alcohol Use and Older Adults ?",
    "What causes Osteoarthritis ?",
    "How many people are affected by Osteoarthritis ?",
    "What are the treatments for Prostate Cancer ?",
    "What are the symptoms of Poikiloderma with neutropenia ?",
    "Is ornithine translocase deficiency inherited ?"
]

def load_medquad():
    df = pd.read_csv(RAW_CSV)
    # Build a simple lookup from question text to original answer
    # (exact match; for more precise mapping you could use IDs if available)
    lookup = {}
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a and q not in lookup:
            lookup[q] = a
    return lookup

def main():
    medquad_lookup = load_medquad()

    rows = []
    for q in TEST_QUESTIONS:
        print("Question:", q)

        # RAG answer
        rag_resp = requests.post(API_URL, json={"question": q}, timeout=60)
        rag_resp.raise_for_status()
        rag_data = rag_resp.json()
        rag_answer = rag_data.get("answer", "")

        rag_grade = textstat.flesch_kincaid_grade(rag_answer)

        # Original MedQuAD answer: try exact question match; otherwise leave blank
        orig_answer = medquad_lookup.get(q, "")
        orig_grade = textstat.flesch_kincaid_grade(orig_answer) if orig_answer else None

        print(f"  RAG grade: {rag_grade:.2f}, Orig grade: {orig_grade if orig_grade is not None else 'N/A'}")

        rows.append({
            "question": q,
            "rag_answer": rag_answer,
            "rag_flesch_kincaid_grade": rag_grade,
            "orig_answer": orig_answer,
            "orig_flesch_kincaid_grade": orig_grade if orig_grade is not None else "",
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "rag_answer",
                "rag_flesch_kincaid_grade",
                "orig_answer",
                "orig_flesch_kincaid_grade",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote evaluation with baselines to", OUT_CSV)

if __name__ == "__main__":
    main()
