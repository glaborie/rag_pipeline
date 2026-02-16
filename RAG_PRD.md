Here is a comprehensive Product Requirements Document (PRD) and a corresponding Test Suite strategy for a Python-based RAG (Retrieval-Augmented Generation) API.

---

# Part 1: Product Requirements Document (PRD)

**Project Name:** DocuQuery RAG API
**Version:** 1.0
**Status:** Draft
**Language:** Python 3.10+

## 1. Executive Summary
The DocuQuery API is a backend service that allows users to upload multiple PDF documents and perform natural language queries against the content of those documents. The system utilizes a Vector Database to store semantic embeddings. Crucially, the system supports dynamic indexing, ensuring that the knowledge base is updated immediately when new documents are uploaded without requiring system downtime.

## 2. Problem Statement
Users have large repositories of PDF documentation but cannot easily search for specific answers or summaries across multiple files. Keyword search is insufficient; semantic understanding is required.

## 3. User Stories
*   **As a Developer**, I want an API endpoint to upload PDF files so that they are added to the knowledge base.
*   **As a User**, I want to ask a question in natural language and receive an answer based strictly on the uploaded PDFs.
*   **As a User**, I want to know which document (source) the answer came from.
*   **As a System Admin**, I want the index to update automatically when a new file is added, avoiding duplicate processing of existing files.

## 4. Functional Requirements

### 4.1. Ingestion Pipeline
*   **FR-01: File Upload:** The API must accept `multipart/form-data` uploads containing one or multiple PDF files.
*   **FR-02: Text Extraction:** The system must extract text from PDFs (handling standard text).
*   **FR-03: Chunking:** Text must be split into manageable chunks (e.g., 1000 tokens with 200 token overlap) to optimize retrieval.
*   **FR-04: Vectorization:** Chunks must be converted into vector embeddings using an embedding model (e.g., OpenAI `text-embedding-3-small` or HuggingFace local models).
*   **FR-05: Index Refresh (Upsert):**
    *   When new files are submitted, the system must process only the new files.
    *   Old embeddings for the same filename should be overwritten or versioned (Idempotency).
    *   The vector store must persist data between restarts.

### 4.2. Retrieval & Generation
*   **FR-06: Query Interface:** An endpoint accepting a JSON payload with a text question.
*   **FR-07: Semantic Search:** The system must retrieve the top $K$ (e.g., 4) most relevant chunks from the vector store based on cosine similarity.
*   **FR-08: LLM Synthesis:** The retrieved chunks + the user query must be sent to an LLM (e.g., GPT-4o or Llama 3) to generate a coherent answer.
*   **FR-09: Source Citation:** The response must include metadata indicating which source file(s) were used.

### 4.3. Management
*   **FR-10: List Documents:** An endpoint to retrieve a list of currently indexed filenames.
*   **FR-11: Clear Index:** An endpoint to wipe the database (for testing/reset).

## 5. Technical Architecture

### 5.1. Tech Stack
*   **Language:** Python 3.10+
*   **Web Framework:** FastAPI (for high-performance, async, and auto-swagger docs).
*   **Orchestration:** LangChain or LlamaIndex.
*   **Vector Database:** ChromaDB (Local/Persistent) or Qdrant.
*   **LLM/Embeddings:** OpenAI API (Standard) or Ollama (Local).
*   **PDF Parsing:** `pypdf` or `unstructured`.

### 5.2. Data Flow
1.  **POST /ingest**: PDF -> Text Loader -> Splitter -> Embedding Model -> **Vector Store**.
2.  **POST /query**: Question -> Embedding Model -> Vector Search (Context) -> LLM (Prompt + Context) -> **Response**.

## 6. API Specifications

### Endpoint 1: Ingest Documents
*   **Path:** `POST /api/v1/ingest`
*   **Body:** `files: List[UploadFile]`
*   **Behavior:** Parses PDFs, creates embeddings, updates index.
*   **Response:** `200 OK` `{ "message": "Successfully indexed 2 files.", "files": ["doc1.pdf", "doc2.pdf"] }`

### Endpoint 2: Chat / Query
*   **Path:** `POST /api/v1/chat`
*   **Body:** `{ "query": "What is the conclusion of the audit report?" }`
*   **Response:**
    ```json
    {
      "answer": "The audit report concludes that...",
      "sources": ["audit_2024.pdf (Page 4)"]
    }
    ```

### Endpoint 3: List Documents
*   **Path:** `GET /api/v1/documents`
*   **Response:** `{ "documents": ["audit_2024.pdf", "specs.pdf"] }`

---

# Part 2: Implementation & Full Test Suite

Below is the structure and code for the test suite. We will use **Pytest** and **FastAPI TestClient**.

**Assumptions for this code:**
*   You are using `LangChain`, `FastAPI`, and `ChromaDB`.
*   You have an OpenAI API Key (or we will mock it for tests).

## Directory Structure
```text
/rag_project
├── app/
│   ├── main.py          # FastAPI Entry point
│   ├── core.py          # RAG Logic (Indexing/Chat)
│   └── db.py            # Vector DB setup
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Fixtures (PDF creation, Mocks)
│   ├── test_api.py      # Integration tests for Endpoints
│   └── test_logic.py    # Unit tests for chunking/cleaning
├── requirements.txt
└── pytest.ini
```

## 1. The Application Code (Simplified for Context)

To make the tests make sense, here is the minimal logic required in `app/core.py`:

```python
# app/core.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Mockable global for the vector store
VECTOR_STORE = None

def get_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        # In a real app, use persistent directory
        VECTOR_STORE = Chroma(embedding_function=OpenAIEmbeddings())
    return VECTOR_STORE

async def process_documents(files: List[str]):
    """
    1. Load PDFs
    2. Split Text
    3. Index (Refresh)
    """
    documents = []
    for file_path in files:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    db = get_vector_store()
    # Logic to add documents (Chroma handles ID management automatically if IDs provided, 
    # but for simplicity we just add here)
    db.add_documents(splits)
    return len(splits)

def query_rag(query_text: str):
    db = get_vector_store()
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": query_text})
    return {
        "answer": result["result"],
        "sources": list(set([doc.metadata.get("source") for doc in result["source_documents"]]))
    }
```

## 2. The Test Suite

### `tests/conftest.py` (Fixtures)
This file handles creating temporary PDFs and mocking OpenAI so we don't spend money running tests.

```python
import pytest
import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

# Import the app (assuming app.main has the 'app' object)
# form app.main import app 

@pytest.fixture(scope="function")
def mock_openai():
    """Mocks OpenAI Embeddings and LLM calls."""
    with patch("app.core.OpenAIEmbeddings") as mock_embed, \
         patch("app.core.ChatOpenAI") as mock_llm:
        
        # Mock Embeddings
        mock_instance = mock_embed.return_value
        # Return a fake vector of length 1536 (standard openai size)
        mock_instance.embed_documents.return_value = [[0.1] * 1536] 
        mock_instance.embed_query.return_value = [0.1] * 1536
        
        # Mock LLM Response
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.invoke.return_value.content = "This is a mocked answer."
        
        yield mock_embed, mock_llm

@pytest.fixture(scope="session")
def test_pdf_path(tmp_path_factory):
    """Creates a dummy PDF file for testing ingestion."""
    fn = tmp_path_factory.mktemp("data") / "test_doc.pdf"
    c = canvas.Canvas(str(fn))
    c.drawString(100, 750, "This is a test PDF document regarding Project Alpha.")
    c.drawString(100, 730, "The secret code is 12345.")
    c.save()
    return str(fn)

@pytest.fixture(scope="function")
def mock_vector_store():
    """Mocks the Chroma Vector Store to avoid disk I/O."""
    with patch("app.core.Chroma") as mock_chroma:
        mock_db = MagicMock()
        mock_chroma.return_value = mock_db
        
        # Mock retriever behavior
        mock_retriever = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever
        
        yield mock_db
```

### `tests/test_api.py` (Integration Tests)
Tests the HTTP endpoints.

```python
import pytest
from unittest.mock import patch, MagicMock

# Assuming your FastAPI app is defined in app.main
# from app.main import app
# client = TestClient(app)

# --- MOCK APP FOR DEMONSTRATION ---
from fastapi import FastAPI, UploadFile, File
from app.core import process_documents, query_rag
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/api/v1/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    # Save files momentarily (logic simplified for test)
    temp_names = []
    for file in files:
        temp_names.append(file.filename)
    
    # Call core logic
    # In real app, save bytes to disk, then pass path
    # Here we mock the processing
    await process_documents(temp_names) 
    return {"message": f"Indexed {len(files)} files"}

@app.post("/api/v1/chat")
def chat(payload: Query):
    return query_rag(payload.query)
# ----------------------------------

client = TestClient(app)

@pytest.mark.asyncio
async def test_ingest_endpoint(mock_openai, mock_vector_store):
    """Test uploading a file triggers the indexer."""
    
    # Mock the internal logic that reads files from disk
    with patch("app.core.PyPDFLoader") as MockLoader:
        # Setup Mock Loader to return a dummy document
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.pdf"}
        MockLoader.return_value.load.return_value = [mock_doc]

        # Create a dummy file payload
        files = [
            ('files', ('test.pdf', b'PDF_CONTENT', 'application/pdf'))
        ]
        
        response = client.post("/api/v1/ingest", files=files)
        
        assert response.status_code == 200
        assert "Indexed 1 files" in response.json()['message']
        
        # Verify vector store add_documents was called
        mock_vector_store.add_documents.assert_called()

def test_chat_endpoint(mock_openai, mock_vector_store):
    """Test the chat endpoint returns an answer and sources."""
    
    # Mock the RetrievalQA chain behavior
    with patch("app.core.RetrievalQA") as MockQA:
        mock_chain = MockQA.from_chain_type.return_value
        
        # Mock the result of the chain
        mock_source_doc = MagicMock()
        mock_source_doc.metadata = {"source": "test.pdf"}
        
        mock_chain.invoke.return_value = {
            "result": "The secret code is 12345.",
            "source_documents": [mock_source_doc]
        }
        
        payload = {"query": "What is the secret code?"}
        response = client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The secret code is 12345."
        assert "test.pdf" in data["sources"]
```

### `tests/test_logic.py` (Unit Tests)
Tests the specific RAG components (Parsing, Splitter).

```python
import pytest
from app.core import process_documents
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_document_indexing_flow(mock_openai, mock_vector_store, test_pdf_path):
    """
    Test the specific logic of reading a PDF, splitting it, 
    and sending it to the vector store.
    """
    
    # Use the real PyPDFLoader here to test actual parsing
    # But mock the VectorStore so we don't need a real DB running
    
    count = await process_documents([test_pdf_path])
    
    # Assertions
    # 1. Check that text was extracted (PyPDFLoader works)
    # 2. Check that RecursiveCharacterTextSplitter worked (should produce at least 1 chunk)
    assert count >= 1
    
    # 3. Verify interaction with Vector DB
    # Ensure add_documents was called on the mock object
    mock_vector_store.add_documents.assert_called_once()
    
    # Inspect arguments passed to add_documents
    call_args = mock_vector_store.add_documents.call_args[0][0] # The list of docs
    assert len(call_args) > 0
    assert "Project Alpha" in call_args[0].page_content
```

## 3. Running the Tests

To run this suite, the user would execute:

```bash
# Install dependencies
pip install fastapi pytest httpx langchain langchain-openai langchain-chroma pypdf reportlab

# Run tests
pytest -v
```

## 4. Test Strategy Summary

| Test Type | Scope | Success Criteria |
| :--- | :--- | :--- |
| **Unit** | `PyPDFLoader`, Text Splitter | PDF text is extracted accurately; Text is split into chunks < 1000 tokens. |
| **Integration** | `POST /ingest` | 200 OK; Vector DB `add_documents` method is triggered. |
| **Integration** | `POST /chat` | 200 OK; JSON response contains "answer" and "sources". |
| **E2E (Mocked)** | Full Flow | Upload -> Index -> Query -> Correct Mock Response returned. |
| **Refresh Logic** | Indexing | Uploading a new file adds to the index without clearing previous data (verified by `add_documents` calls). |