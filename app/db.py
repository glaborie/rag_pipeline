import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_VECTOR_STORE: Optional[Chroma] = None


def _openrouter_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _openrouter_api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _openrouter_headers() -> dict:
    headers = {}
    app_name = os.environ.get("OPENROUTER_APP_NAME")
    app_url = os.environ.get("OPENROUTER_APP_URL")
    if app_name:
        headers["HTTP-Referer"] = app_url or "http://localhost"
        headers["X-Title"] = app_name
    return headers


def get_persist_directory() -> str:
    return os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")


def get_vector_store() -> Chroma:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        persist_dir = get_persist_directory()
        os.makedirs(persist_dir, exist_ok=True)
        print(
            f"[DEBUG] Creating OpenAIEmbeddings with model={os.environ.get('OPENROUTER_EMBED_MODEL', 'text-embedding-3-small')}, base_url={_openrouter_base_url()}, api_key={_openrouter_api_key()}, default_headers={_openrouter_headers()}")
        _VECTOR_STORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(
                model=os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"),
                base_url=_openrouter_base_url(),
                api_key=_openrouter_api_key(),
                default_headers=_openrouter_headers(),
            ),
        )
    return _VECTOR_STORE


def reset_vector_store() -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = None
