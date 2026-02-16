# DocuQuery RAG API

A Python FastAPI service that ingests PDF documents, stores embeddings in a persistent Chroma vector store, and answers natural language questions with source citations.

## Features

- Upload one or multiple PDFs for indexing
- Automatic chunking and embedding
- Semantic search + LLM synthesis
- Source citations in responses
- Persistent vector store (Chroma)

## Requirements

- Python 3.10+
- An OpenRouter API key for embeddings and chat

## Setup

1. Create a virtual environment.
2. Install dependencies from requirements.txt.
3. Set the OpenRouter API key:
  - OPENROUTER_API_KEY

Optional environment variables:

- OPENROUTER_CHAT_MODEL (default: openai/gpt-4o-mini)
- OPENROUTER_EMBED_MODEL (default: text-embedding-3-small)
- OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
- OPENROUTER_APP_NAME (sets OpenRouter X-Title header)
- OPENROUTER_APP_URL (sets OpenRouter HTTP-Referer header)
- CHROMA_PERSIST_DIR (default: ./data/chroma)
- UPLOAD_DIR (default: ./data/uploads)

## Run the API

Use any ASGI server (for example, uvicorn) to run app.main:app.

OpenAPI docs are available at /docs.

## API Endpoints

- POST /api/v1/ingest
  - multipart/form-data with one or more files
- POST /api/v1/chat
  - JSON: { "query": "..." }
- GET /api/v1/documents
  - List indexed documents
- DELETE /api/v1/index
  - Clear the index (testing/reset)

## Tests

Run the test suite with pytest. The tests mock external calls and do not require a real OpenRouter key.

## Project Structure

- app/main.py: FastAPI application and routes
- app/core.py: RAG logic (ingestion, retrieval, generation)
- app/db.py: Vector store setup
- tests/: pytest test suite
