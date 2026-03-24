"""
Application configuration module.

Centralizes all configurable parameters, supporting both local .env
and Streamlit Cloud Secrets for seamless deployment across environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import streamlit as st
from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    """LLM and embedding model configuration."""

    embedding_model: str = "text-embedding-v2"
    llm_model: str = "qwen-turbo"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.2
    max_tokens: int = 1500


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-Augmented Generation pipeline configuration."""

    chunk_size: int = 500
    chunk_overlap: int = 100
    retriever_top_k: int = 3
    separators: tuple[str, ...] = (
        "\n\n", "\n", "。", "！", "？", "；", "，", " ",
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    page_title: str = "AI 智能学习助手"
    page_icon: str = "🤖"
    vector_store_path: str = "./saved_vector_store"
    supported_file_types: tuple[str, ...] = ("txt", "pdf")
    txt_encoding_fallbacks: tuple[str, ...] = ("utf-8", "gb18030", "gbk")

    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


def resolve_api_key() -> str:
    """
    Resolve the DashScope API key with priority:
      1. Streamlit Secrets (cloud deployment)
      2. Environment variable via .env (local development)

    Raises:
        SystemExit: If no API key is found, stops the Streamlit app.
    """
    load_dotenv()

    api_key: str | None = None
    try:
        api_key = st.secrets["DASHSCOPE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        st.error(
            "❌ 未检测到 `DASHSCOPE_API_KEY`！\n\n"
            "- **本地运行**：请在 `.env` 文件中配置\n"
            "- **线上部署**：请在 Streamlit Secrets 中配置"
        )
        st.stop()

    os.environ["DASHSCOPE_API_KEY"] = api_key
    return api_key
