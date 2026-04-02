"""
向量库生命周期管理模块。

通过 Streamlit 会话状态封装 FAISS 索引的创建、持久化与加载流程。
"""

from __future__ import annotations

import logging
import os
import shutil
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
    管理 FAISS 向量库的完整生命周期。

    职责：
      - 基于文档构建索引
      - 索引落盘与加载
      - 暴露检索器接口
    """

    def __init__(self, embeddings: Embeddings, persist_path: str) -> None:
        self._embeddings = embeddings
        self._persist_path = persist_path

    # ---- 对外接口 ----

    @property
    def store(self) -> FAISS | None:
        """当前内存中的向量库对象（不存在时为 ``None``）。"""
        return cast(FAISS | None, st.session_state.get(_SESSION_KEY))

    @property
    def is_ready(self) -> bool:
        """会话中是否存在可用向量库。"""
        return self.store is not None

    def build_from_documents(self, documents: list[Document]) -> FAISS:
        """
        基于 *documents* 构建新的 FAISS 索引，落盘后写入会话状态。
        """
        store = FAISS.from_documents(documents, self._embeddings)
        store.save_local(self._persist_path)
        st.session_state[_SESSION_KEY] = store
        logger.info(
            "已构建向量库：%d 个文档，保存路径 %s",
            len(documents), self._persist_path,
        )
        return store

    def try_load_persisted(self) -> bool:
        """
        尝试加载历史持久化索引。

        返回：
            加载成功返回 ``True``，否则返回 ``False``。
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
            logger.info("已从 %s 加载历史向量库", self._persist_path)
            st.toast("✅ 已加载历史知识库", icon="📚")
            return True
        except Exception:
            logger.exception("加载历史向量库失败")
            st.session_state[_SESSION_KEY] = None
            return False

    def clear(self) -> bool:
        """
        从会话与磁盘中移除当前向量库。

        返回：
            若会话或磁盘中原本存在向量库则返回 ``True``，否则返回 ``False``。
        """
        had_store = self.is_ready or os.path.exists(self._persist_path)
        st.session_state[_SESSION_KEY] = None

        if not os.path.exists(self._persist_path):
            return had_store

        try:
            if os.path.isdir(self._persist_path):
                shutil.rmtree(self._persist_path)
            else:
                os.remove(self._persist_path)
            logger.info("已清理历史向量库：%s", self._persist_path)
            return had_store
        except Exception as exc:
            logger.exception("清理历史向量库失败")
            raise RuntimeError("清空知识库失败，请稍后重试。") from exc

    def get_retriever(self, top_k: int = 3) -> BaseRetriever:
        """
        返回基于当前向量库的 LangChain 检索器。

        异常：
            RuntimeError：当前无可用向量库时抛出。
        """
        store = self.store
        if store is None:
            raise RuntimeError("向量库尚未初始化，请先上传文档。")
        return store.as_retriever(search_kwargs={"k": top_k})
