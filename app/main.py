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
    errors: List[str] = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(file_path))

    try:
        count = await process_documents(saved_paths)
    except Exception as e:
        errors.append(str(e))
        count = 0

    response = {
        "message": f"Successfully indexed {len(files)} files.",
        "files": [Path(p).name for p in saved_paths],
        "chunks": count,
    }
    if errors:
        response["errors"] = errors
    return response


@app.post("/api/v1/chat")
def chat(payload: QueryPayload):
    try:
        result = query_rag(payload.query)
    except Exception as e:
        result = {"answer": "", "sources": [], "error": str(e)}
    return result


@app.get("/api/v1/documents")
def documents():
    return {"documents": list_documents()}


@app.delete("/api/v1/index")
def wipe_index():
    clear_index()
    return {"message": "Index cleared."}
