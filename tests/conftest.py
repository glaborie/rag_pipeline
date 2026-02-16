import os

import pytest
from unittest.mock import MagicMock, patch
from reportlab.pdfgen import canvas


@pytest.fixture(scope="function")
def mock_openai():
    with patch("app.db.OpenAIEmbeddings") as mock_embed, \
         patch("app.core.ChatOpenAI") as mock_llm, \
         patch("app.db.Chroma") as mock_chroma:
        mock_db = MagicMock()
        mock_chroma.return_value = mock_db
        mock_instance = mock_embed.return_value
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_instance.embed_query.return_value = [0.1] * 1536

        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.invoke.return_value.content = "This is a mocked answer."

        mock_retriever = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever
        mock_db.get.return_value = {"metadatas": []}

        with patch("app.db.get_vector_store", return_value=mock_db):
            yield mock_embed, mock_llm, mock_db


@pytest.fixture(scope="session")
def test_pdf_path(tmp_path_factory):
    custom_path = os.environ.get("TEST_PDF_PATH")
    if custom_path and os.path.exists(custom_path):
        return custom_path

    fn = tmp_path_factory.mktemp("data") / "test_doc.pdf"
    c = canvas.Canvas(str(fn))
    c.drawString(100, 750, "This is a test PDF document regarding Project Alpha.")
    c.drawString(100, 730, "The secret code is 12345.")
    c.save()
    return str(fn)


@pytest.fixture(scope="function")
def mock_vector_store(mock_openai):
    # mock_openai yields (mock_embed, mock_llm, mock_db)
    return mock_openai[2]
