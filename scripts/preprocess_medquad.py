# TODO: implement MedQuAD preprocessing
import pandas as pd
from pathlib import Path
import nltk

# Download sentence tokenizer the first time
nltk.download("punkt")
nltk.download('punkt_tab')

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RAW_DIR / "medquad_raw.csv"   # adjust name if different
OUTPUT_CSV = PROC_DIR / "medquad_chunks.csv"

def chunk_text(text, max_words=120):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(str(text))
    chunks, current, count = [], [], 0
    for s in sentences:
        words = s.split()
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def main():
    print("Loading", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    rows = []
    for i, row in df.iterrows():
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        url = str(row.get("url", ""))
        topic = str(row.get("source", ""))
        if not a.strip():
            continue
        chunks = chunk_text(a)
        for j, ch in enumerate(chunks):
            rows.append({
                "chunk_id": f"{i}_{j}",
                "question": q,
                "answer_chunk": ch,
                "url": url,
                "topic": topic
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Wrote", len(out_df), "chunks to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
