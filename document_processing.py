"""
文档处理模块。

通过策略模式实现文件加载流程，支持在不修改既有逻辑的前提下
扩展新的文件格式。
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
# 策略模式：可插拔文件加载器
# ---------------------------------------------------------------------------

class FileLoaderStrategy(ABC):
    """文档加载策略的抽象基类。"""

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """从 *file_path* 加载文档。"""


class TxtLoaderStrategy(FileLoaderStrategy):
    """
    支持自动编码识别的 TXT 加载器。

    会按顺序尝试常见编码（UTF-8 → GB18030 → GBK），
    以兼容不同平台生成的文本文件。
    """

    def __init__(self, encodings: tuple[str, ...] = ("utf-8", "gb18030", "gbk")) -> None:
        self._encodings = encodings

    def load(self, file_path: str) -> list[Document]:
        last_error: Exception | None = None
        for enc in self._encodings:
            try:
                loader = TextLoader(file_path, encoding=enc)
                docs = loader.load()
                logger.info("使用编码 %s 成功加载 TXT", enc)
                return docs
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.debug("编码 %s 加载失败：%s", enc, exc)
        raise ValueError(
            f"无法加载 TXT 文件，已尝试编码 {self._encodings}。"
            f"最后一次错误：{last_error}"
        )


class PdfLoaderStrategy(FileLoaderStrategy):
    """基于 PyPDFLoader 的 PDF 加载器。"""

    def load(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info("成功加载 PDF，共 %d 页", len(docs))
        return docs


# ---------------------------------------------------------------------------
# 加载器注册表（对扩展开放、对修改关闭）
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, type[FileLoaderStrategy]] = {
    ".txt": TxtLoaderStrategy,
    ".pdf": PdfLoaderStrategy,
}


def register_loader(extension: str, loader_cls: type[FileLoaderStrategy]) -> None:
    """为新文件扩展名注册自定义加载器。"""
    _LOADER_REGISTRY[extension.lower()] = loader_cls


def get_loader(extension: str) -> FileLoaderStrategy:
    """
    根据扩展名获取对应的加载器实例。

    异常：
        ValueError：扩展名不受支持时抛出。
    """
    ext = extension.lower()
    loader_cls = _LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise ValueError(f"不支持的文件格式 '{ext}'。当前支持：{supported}")
    return loader_cls()


# ---------------------------------------------------------------------------
# 核心文档处理流水线
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    端到端文档处理流水线。

    职责：
      1. 将上传文件写入临时路径
      2. 委托给对应策略进行加载
      3. 将文档切分为适合检索的片段
      4. 清理临时文件
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
        将上传文件处理为切分后的文档片段。

        参数：
            uploaded_file: Streamlit 的 ``UploadedFile`` 对象。

        返回：
            可直接用于向量化的 ``Document`` 片段列表。
        """
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        temp_path: str | None = None

        try:
            # 1. 写入临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            # 2. 通过策略加载
            if temp_path is None:
                raise RuntimeError("临时文件创建失败")
            loader = get_loader(suffix)
            docs = loader.load(temp_path)

            # 3. 切分为片段
            chunks = self._splitter.split_documents(docs)
            logger.info(
                "已处理 '%s'：%d 个文档 → %d 个片段",
                uploaded_file.name, len(docs), len(chunks),
            )
            return chunks

        finally:
            # 4. 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as exc:
                    logger.warning("删除临时文件失败 %s：%s", temp_path, exc)
