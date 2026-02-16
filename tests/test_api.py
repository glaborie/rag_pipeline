from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_ingest_endpoint(mock_openai, mock_vector_store, test_pdf_path):
    with patch("app.core.PyPDFLoader") as MockLoader:
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.pdf", "page": 0}
        MockLoader.return_value.load.return_value = [mock_doc]

        files = [("files", ("test.pdf", b"PDF_CONTENT", "application/pdf"))]
        response = client.post("/api/v1/ingest", files=files)

        assert response.status_code == 200
        assert "Successfully indexed 1 files" in response.json()["message"]
        mock_vector_store.add_documents.assert_called()


def test_chat_endpoint(mock_openai, mock_vector_store):
    with patch("app.core.RetrievalQA") as MockQA:
        mock_chain = MockQA.from_chain_type.return_value

        mock_source_doc = MagicMock()
        mock_source_doc.metadata = {"source": "test.pdf", "page": 0}

        mock_chain.invoke.return_value = {
            "result": "The secret code is 12345.",
            "source_documents": [mock_source_doc],
        }

        payload = {"query": "What is the secret code?"}
        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The secret code is 12345."
        assert any("test.pdf" in s for s in data["sources"])
