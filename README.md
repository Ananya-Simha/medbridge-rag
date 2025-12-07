\# MedBridge: Medical RAG for Patient-Friendly Answers



\## Overview

MedBridge is a Retrieval Augmented Generation (RAG) system that uses the MedQuAD dataset to generate layperson-friendly, cited medical answers. \[file:1]



\## Project structure

\- `api/` – FastAPI backend with RAG pipeline.

\- `app/` – Streamlit frontend.

\- `scripts/` – Data download, preprocessing, indexing, evaluation.

\- `data/` – Raw, processed data, and vector index.

\- `models/` – Model configuration notes.

\- `Dockerfile` – Containerization.



\## Quickstart

## Setup (without Docker)

1. Clone the repo:

- git clone https://github.com/Ananya-Simha/medbridge-rag.git
- cd medbridge-rag
- python -m venv .venv
- source .venv/bin/activate # .venv\Scripts\activate on Windows
- pip install -r requirements.txt

2. Obtain MedQuAD:

- Clone https://github.com/abachaa/MedQuAD and run `to_csv.py` to produce `medquad_raw.csv` with question, answer, url, source. [file:42]
- Copy `medquad_raw.csv` into `data/raw/medquad_raw.csv`.

3. Build chunks and index:

- python scripts/preprocess_medquad.py
- python scripts/build_index.py

4. Run backend:

- uvicorn api.main:app --host 0.0.0.0 --port 8000


5. Run Streamlit UI (locally):

- streamlit run app/streamlit_app.py

- Ensure `API_URL` in `app/streamlit_app.py` points to your backend.


## Setup with Docker

1. Build the image:
- docker build -t medbridge-rag .

2. Ensure `data/raw/medquad_raw.csv` is available on the host.

3. Run the container, mounting the data directory:
- docker run -p 8000:8000 -v $(pwd)/data:/app/data medbridge-rag

This will:
- Preprocess MedQuAD into chunks if needed.
- Build the FAISS index if needed.
- Start the FastAPI API at `http://localhost:8000/answer`.



