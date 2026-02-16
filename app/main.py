import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.core import clear_index, list_documents, process_documents, query_rag

app = FastAPI(title="DocuQuery RAG API", version="1.0")

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryPayload(BaseModel):
    query: str


@app.post("/api/v1/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    saved_paths: List[str] = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(file_path))

    count = await process_documents(saved_paths)
    return {
        "message": f"Successfully indexed {len(files)} files.",
        "files": [Path(p).name for p in saved_paths],
        "chunks": count,
    }


@app.post("/api/v1/chat")
def chat(payload: QueryPayload):
    return query_rag(payload.query)


@app.get("/api/v1/documents")
def documents():
    return {"documents": list_documents()}


@app.delete("/api/v1/index")
def wipe_index():
    clear_index()
    return {"message": "Index cleared."}
