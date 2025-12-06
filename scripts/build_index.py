# TODO: implement index building
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROC_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_CSV = PROC_DIR / "medquad_chunks.csv"
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
META_FILE = INDEX_DIR / "metadata.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Loading chunks from", CHUNKS_CSV)
    df = pd.read_csv(CHUNKS_CSV)
    texts = df["answer_chunk"].astype(str).tolist()
    print("Encoding", len(texts), "chunks...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    dim = embeddings.shape[1]
    # Normalize for inner product cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    df.to_parquet(META_FILE, index=False)
    print("Saved index to", INDEX_FILE)
    print("Saved metadata to", META_FILE)

if __name__ == "__main__":
    main()
