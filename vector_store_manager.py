"""
Vector store lifecycle management.

Encapsulates FAISS index creation, persistence, and loading with
thread-safe singleton semantics via Streamlit session state.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

import streamlit as st
from langchain_community.vectorstores import FAISS

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

_SESSION_KEY = "vector_store"


class VectorStoreManager:
    """
    Manages the full lifecycle of the FAISS vector store.

    Responsibilities:
      - Build index from documents
      - Persist / load from disk
      - Expose a retriever interface
    """

    def __init__(self, embeddings: Embeddings, persist_path: str) -> None:
        self._embeddings = embeddings
        self._persist_path = persist_path

    # ---- public API ----

    @property
    def store(self) -> FAISS | None:
        """Current in-memory vector store (or ``None``)."""
        return cast(FAISS | None, st.session_state.get(_SESSION_KEY))

    @property
    def is_ready(self) -> bool:
        """Whether a usable vector store exists in session."""
        return self.store is not None

    def build_from_documents(self, documents: list[Document]) -> FAISS:
        """
        Build a new FAISS index from *documents*, persist to disk,
        and store in session state.
        """
        store = FAISS.from_documents(documents, self._embeddings)
        store.save_local(self._persist_path)
        st.session_state[_SESSION_KEY] = store
        logger.info(
            "Built vector store with %d documents, saved to %s",
            len(documents), self._persist_path,
        )
        return store

    def try_load_persisted(self) -> bool:
        """
        Attempt to load a previously persisted index.

        Returns:
            ``True`` if loaded successfully, ``False`` otherwise.
        """
        if _SESSION_KEY in st.session_state:
            return self.is_ready

        if not os.path.exists(self._persist_path):
            st.session_state[_SESSION_KEY] = None
            return False

        try:
            store = FAISS.load_local(
                self._persist_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            st.session_state[_SESSION_KEY] = store
            logger.info("Loaded persisted vector store from %s", self._persist_path)
            st.toast("✅ 已加载历史知识库", icon="📚")
            return True
        except Exception:
            logger.exception("Failed to load persisted vector store")
            st.session_state[_SESSION_KEY] = None
            return False

    def get_retriever(self, top_k: int = 3) -> BaseRetriever:
        """
        Return a LangChain retriever backed by the current store.

        Raises:
            RuntimeError: If no vector store is available.
        """
        store = self.store
        if store is None:
            raise RuntimeError("向量库尚未初始化，请先上传文档。")
        return store.as_retriever(search_kwargs={"k": top_k})
