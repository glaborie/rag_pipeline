import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="function")
def client(mock_vector_store):
    with patch("app.db.get_vector_store", return_value=mock_vector_store):
        yield TestClient(app)

def test_documents_endpoint_empty(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json()["documents"] == []

def test_wipe_index_endpoint(client):
    response = client.delete("/api/v1/index")
    assert response.status_code == 200
    assert "Index cleared" in response.json()["message"]

@pytest.mark.asyncio
def test_ingest_multiple_files(mock_openai, mock_vector_store, test_pdf_path):
    from unittest.mock import patch, MagicMock
    with patch("app.core.PyPDFLoader") as MockLoader:
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Test content 1"
        mock_doc1.metadata = {"source": "test1.pdf", "page": 0}
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Test content 2"
        mock_doc2.metadata = {"source": "test2.pdf", "page": 0}
        MockLoader.return_value.load.side_effect = [[mock_doc1], [mock_doc2]]

        files = [
            ("files", ("test1.pdf", b"PDF_CONTENT_1", "application/pdf")),
            ("files", ("test2.pdf", b"PDF_CONTENT_2", "application/pdf")),
        ]
        with patch("app.db.get_vector_store", return_value=mock_vector_store):
            test_client = TestClient(app)
            response = test_client.post("/api/v1/ingest", files=files)
            assert response.status_code == 200
            assert "Successfully indexed 2 files" in response.json()["message"]
            assert response.json()["chunks"] == 2

@pytest.mark.asyncio
def test_ingest_invalid_file_type(mock_openai, mock_vector_store):
    files = [
        ("files", ("not_a_pdf.txt", b"Just text", "text/plain")),
    ]
    with patch("app.db.get_vector_store", return_value=mock_vector_store):
        test_client = TestClient(app)
        response = test_client.post("/api/v1/ingest", files=files)
        assert response.status_code == 200
        # Should still index, but content may be empty or error handled
        assert "Successfully indexed 1 files" in response.json()["message"]

@pytest.mark.asyncio
def test_chat_no_documents(mock_openai, mock_vector_store):
    payload = {"query": "What is the secret code?"}
    with patch("app.db.get_vector_store", return_value=mock_vector_store):
        test_client = TestClient(app)
        response = test_client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["sources"] == []
