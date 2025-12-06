# TODO: implement retriever here
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
META_FILE = INDEX_DIR / "metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, k=5):
        self.k = k
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_FILE))
        self.meta = pd.read_parquet(META_FILE)

    def retrieve(self, query: str):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, self.k)
        idxs, scores = idxs[0], scores[0]
        results = []
        for i, s in zip(idxs, scores):
            row = self.meta.iloc[int(i)]
            results.append({
                "chunk_id": row["chunk_id"],
                "answer_chunk": row["answer_chunk"],
                "question": row["question"],
                "url": row["url"],
                "topic": row["topic"],
                "score": float(s)
            })
        return results

if __name__ == "__main__":
    # Quick CLI test
    r = Retriever(k=3)
    q = "What are the symptoms of diabetes?"
    hits = r.retrieve(q)
    for h in hits:
        print("Score:", h["score"])
        print("URL:", h["url"])
        print("Chunk:", h["answer_chunk"][:200], "...\n")
