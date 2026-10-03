"""
FastAPI entry point for the Metadata Extraction API.
TODO: Implement REST endpoints in Phase 2.
"""
from fastapi import FastAPI

app = FastAPI(
    title="Document Metadata Extraction API",
    description="Extracts structured metadata from rental agreement documents (DOCX / PNG / JPG).",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
