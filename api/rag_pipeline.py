# TODO: implement RAG pipeline here
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .retriever import Retriever

# You can change this later based on what runs on your machine
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

class RAGPipeline:
    def __init__(self, device=None, k: int = 5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = Retriever(k=k)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def build_prompt(self, question: str, passages):
        blocks = []
        for i, p in enumerate(passages, start=1):
            blocks.append(f"[{i}] {p['answer_chunk']}\n(Source: {p['url']})")
        context = "\n\n".join(blocks)

        prompt = (
            "You are a medical assistant. You are given a patient question and evidence.\n\n"
            f"Question: {question}\n\n"
            "Evidence:\n"
            f"{context}\n\n"
            "Write a clear, patient-friendly answer using simple language (around 8th grade level). "
            "Only use facts from the evidence. Cite sources inline like [1], [2], etc. "
            "Do not invent new medical facts.\n\n"
            "Answer:\n"
        )
        return prompt

    def generate(self, question: str) -> Dict:
        passages = self.retriever.retrieve(question)
        prompt = self.build_prompt(question, passages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                top_p=0.9
            )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # naive split to get the answer part
        answer = full_text.split("Answer:", 1)[-1].strip()
        return {
            "answer": answer,
            "passages": passages
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    q = "What are the complications of high blood pressure?"
    out = rag.generate(q)
    print("ANSWER:\n", out["answer"][:500])
