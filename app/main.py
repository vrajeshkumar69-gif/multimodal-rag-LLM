from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict
import fitz  # PyMuPDF
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.llm import generate_llm_answer

app = FastAPI(
    title="Multimodal SOP RAG System (LLM Version)"
)

DOCUMENT_STORE: List[Dict] = []
VECTOR_METADATA: List[Dict] = []

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384
VECTOR_INDEX = faiss.IndexFlatL2(VECTOR_DIM)

START_TIME = time.time()


def rebuild_index():
    VECTOR_INDEX.reset()
    texts = []

    for chunk in DOCUMENT_STORE:
        texts.append(chunk["content"])
        VECTOR_METADATA.append({
            "page": chunk["page"],
            "type": chunk["type"],
            "source": chunk["source"]
        })

    if texts:
        embeddings = EMBEDDING_MODEL.encode(texts)
        VECTOR_INDEX.add(np.array(embeddings).astype("float32"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": len(VECTOR_METADATA),
        "uptime_seconds": int(time.time() - START_TIME)
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    DOCUMENT_STORE.clear()
    VECTOR_METADATA.clear()

    for page_no, page in enumerate(doc, start=1):

        text = page.get_text().strip()
        if text:
            DOCUMENT_STORE.append({
                "type": "text",
                "content": text,
                "page": page_no,
                "source": file.filename
            })

        for _ in page.get_images(full=True):
            DOCUMENT_STORE.append({
                "type": "image",
                "content": f"Image on page {page_no} showing diagrams or handling instructions.",
                "page": page_no,
                "source": file.filename
            })

    rebuild_index()

    return {
        "filename": file.filename,
        "indexed_chunks": len(DOCUMENT_STORE)
    }


@app.post("/query")
def query(payload: Dict[str, str]):
    question = payload["question"]

    query_embedding = EMBEDDING_MODEL.encode([question])
    _, indices = VECTOR_INDEX.search(
        np.array(query_embedding).astype("float32"),
        k=min(5, VECTOR_INDEX.ntotal)
    )

    context = []
    sources = []

    for idx in indices:
        chunk = DOCUMENT_STORE[idx]
        context.append(chunk["content"])
        sources.append({
            "page": chunk["page"],
            "type": chunk["type"],
            "source": chunk["source"]
        })

    answer = generate_llm_answer(question, context)

    return {
        "answer": answer,
        "sources": sources
    }
