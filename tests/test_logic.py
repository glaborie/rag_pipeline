import pytest

from app.core import process_documents


@pytest.mark.asyncio
async def test_document_indexing_flow(mock_openai, mock_vector_store, test_pdf_path):
    count = await process_documents([test_pdf_path])

    assert count >= 1
    mock_vector_store.add_documents.assert_called_once()

    call_args = mock_vector_store.add_documents.call_args[0][0]
    assert len(call_args) > 0
    assert "Project Alpha" in call_args[0].page_content
