import hashlib
import os
from typing import Iterable, List, Tuple

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from app.db import get_vector_store


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_doc_id(source: str, page: int, chunk_text: str) -> str:
    return f"{source}|p{page}|{_hash_text(chunk_text)}"


def _load_documents(file_paths: Iterable[str]):
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} docs from {file_path}")
        for doc in loaded:
            if 'source' not in doc.metadata or not doc.metadata['source']:
                doc.metadata['source'] = file_path
            print(f"[DEBUG] Doc metadata: {doc.metadata}")
        documents.extend(loaded)
    return documents


def _split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def _collect_ids_and_sources(splits) -> Tuple[List[str], List[str]]:
    ids = []
    sources = []
    for doc in splits:
        source = doc.metadata.get("source") or "unknown"
        page = int(doc.metadata.get("page", 0))
        ids.append(_build_doc_id(source, page, doc.page_content))
        sources.append(source)
    return ids, sources


async def process_documents(file_paths: List[str]) -> int:
    print(f"[DEBUG] Ingesting files: {file_paths}")
    documents = _load_documents(file_paths)
    print(f"[DEBUG] Loaded {len(documents)} documents.")
    if documents:
        print(f"[DEBUG] Example document metadata: {documents[0].metadata}")
    splits = _split_documents(documents)
    print(f"[DEBUG] Split into {len(splits)} chunks.")
    if splits:
        print(f"[DEBUG] Example split metadata: {splits[0].metadata}")

    ids, sources = _collect_ids_and_sources(splits)
    print(f"[DEBUG] Collected sources: {sources}")
    # _delete_existing_sources(sources)  # TEMP: Skip deletion to debug Chroma add

    db = get_vector_store()
    try:
        db.add_documents(splits, ids=ids)
    except Exception as e:
        print(f"[ERROR] Exception during db.add_documents: {e}")
        raise
    return len(splits)


def query_rag(query_text: str, top_k: int = DEFAULT_TOP_K) -> dict:
    db = get_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        default_headers={
            **(
                {
                    "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", "http://localhost"),
                    "X-Title": os.environ.get("OPENROUTER_APP_NAME"),
                }
                if os.environ.get("OPENROUTER_APP_NAME")
                else {}
            )
        },
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain.invoke({"query": query_text})
    sources = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        if page is not None:
            sources.append(f"{source} (Page {page + 1})")
        else:
            sources.append(source)

    return {
        "answer": result.get("result", ""),
        "sources": sorted(set(sources)),
    }


def list_documents() -> List[str]:
    db = get_vector_store()
    data = db.get(include=["metadatas"])
    sources = []
    for meta in data.get("metadatas", []) or []:
        if meta and isinstance(meta, dict):
            source = meta.get("source")
            if source:
                sources.append(source)
    return sorted(set(sources))


def clear_index() -> None:
    db = get_vector_store()
    db.delete(where={})
