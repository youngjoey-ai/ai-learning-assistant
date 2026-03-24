"""
Document processing module.

Implements the Strategy pattern for file loading, enabling clean
extension to new file formats without modifying existing code.
"""

from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

    from config import AppConfig

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Pattern: Pluggable file loaders
# ---------------------------------------------------------------------------

class FileLoaderStrategy(ABC):
    """Abstract base class for document loading strategies."""

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Load documents from *file_path*."""


class TxtLoaderStrategy(FileLoaderStrategy):
    """
    TXT loader with automatic encoding detection.

    Attempts a sequence of common encodings (UTF-8 → GB18030 → GBK)
    to gracefully handle files from different platforms.
    """

    def __init__(self, encodings: tuple[str, ...] = ("utf-8", "gb18030", "gbk")) -> None:
        self._encodings = encodings

    def load(self, file_path: str) -> list[Document]:
        last_error: Exception | None = None
        for enc in self._encodings:
            try:
                loader = TextLoader(file_path, encoding=enc)
                docs = loader.load()
                logger.info("Loaded TXT with encoding=%s", enc)
                return docs
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.debug("Encoding %s failed: %s", enc, exc)
        raise ValueError(
            f"无法加载 TXT 文件，已尝试编码 {self._encodings}。"
            f"最后一次错误：{last_error}"
        )


class PdfLoaderStrategy(FileLoaderStrategy):
    """PDF loader backed by PyPDFLoader."""

    def load(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info("Loaded PDF: %d pages", len(docs))
        return docs


# ---------------------------------------------------------------------------
# Loader registry (open for extension, closed for modification)
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, type[FileLoaderStrategy]] = {
    ".txt": TxtLoaderStrategy,
    ".pdf": PdfLoaderStrategy,
}


def register_loader(extension: str, loader_cls: type[FileLoaderStrategy]) -> None:
    """Register a custom loader for a new file extension."""
    _LOADER_REGISTRY[extension.lower()] = loader_cls


def get_loader(extension: str) -> FileLoaderStrategy:
    """
    Retrieve the appropriate loader instance for the given extension.

    Raises:
        ValueError: If the extension is not supported.
    """
    ext = extension.lower()
    loader_cls = _LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise ValueError(f"不支持的文件格式 '{ext}'。当前支持：{supported}")
    return loader_cls()


# ---------------------------------------------------------------------------
# Core document processing pipeline
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    End-to-end document processing pipeline.

    Responsibilities:
      1. Persist uploaded file to a temporary path
      2. Delegate loading to the correct strategy
      3. Split documents into retrieval-friendly chunks
      4. Clean up temporary files
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            separators=list(config.rag.separators),
        )

    def process(self, uploaded_file: UploadedFile) -> list[Document]:
        """
        Process an uploaded file into chunked documents.

        Args:
            uploaded_file: The Streamlit ``UploadedFile`` object.

        Returns:
            A list of ``Document`` chunks ready for embedding.
        """
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        temp_path: str | None = None

        try:
            # 1. Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            # 2. Load via strategy
            if temp_path is None:
                raise RuntimeError("临时文件创建失败")
            loader = get_loader(suffix)
            docs = loader.load(temp_path)

            # 3. Split into chunks
            chunks = self._splitter.split_documents(docs)
            logger.info(
                "Processed '%s': %d docs → %d chunks",
                uploaded_file.name, len(docs), len(chunks),
            )
            return chunks

        finally:
            # 4. Clean up
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as exc:
                    logger.warning("Failed to remove temp file %s: %s", temp_path, exc)
