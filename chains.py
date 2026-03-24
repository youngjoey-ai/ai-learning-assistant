"""
LangChain pipeline construction module.

Provides factory functions that build:
  - RAG chain (LCEL architecture) for knowledge-grounded Q&A
  - ReAct agent (LangGraph) for complex multi-step tasks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
你是一个严格依据知识库回答问题的助手。

【规则】
1. 只允许使用参考文档中的内容回答，禁止编造。
2. 如果参考文档中没有相关内容，直接回复：「知识库中无相关信息」。
3. 回答要简洁、清晰、分点说明。
4. 回答结尾附上信息来源。"""


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Structured result from the RAG chain invocation."""

    answer: str
    source_docs: list[Document]


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a numbered, source-attributed string."""
    parts: list[str] = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"【片段 {idx}】（来源：{source} | 页码：{page}）\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def _response_text(response: Any) -> str:
    """Extract a displayable string from LangChain response objects."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_rag_chain(retriever: VectorStoreRetriever, llm: BaseChatModel):
    """
    Build an LCEL-based RAG chain.

    Architecture::

        question ──► retriever ──► format_docs ──► prompt ──► LLM ──► answer
                  └──────────────────────────────────────────────► source_docs

    Returns:
        A callable that accepts a question string and returns ``RAGResult``.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "【参考文档】\n{context}\n\n【用户问题】\n{question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    def invoke(question: str) -> RAGResult:
        """Run the RAG pipeline and return structured result."""
        # Parallel execution: retrieve docs + generate answer
        source_docs = retriever.invoke(question)
        result = chain.invoke(question)
        answer = _response_text(result)
        return RAGResult(answer=answer, source_docs=source_docs)

    return invoke


# ---------------------------------------------------------------------------
# Agent Chain
# ---------------------------------------------------------------------------

def build_agent(retriever: VectorStoreRetriever, llm: BaseChatModel):
    """
    Build a LangGraph ReAct agent equipped with knowledge-base tools.

    Tools:
      - ``knowledge_query``: Semantic search on the uploaded corpus
      - ``content_summary``: LLM-powered summarisation utility
    """

    def knowledge_query(query: str) -> str:
        """Query the user's uploaded knowledge base for relevant content."""
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中无相关信息"
        return _format_docs(docs)

    def content_summary(text: str) -> str:
        """Summarise the given learning content, highlighting key points."""
        response = llm.invoke(
            f"请简洁总结以下学习内容，突出核心知识点：\n\n{text}"
        )
        return _response_text(response)

    tools = [
        Tool(
            name="knowledge_query",
            func=knowledge_query,
            description=(
                "查询用户上传的知识库。当问题涉及学习笔记、知识点、"
                "项目或总结时，优先使用此工具。"
            ),
        ),
        Tool(
            name="content_summary",
            func=content_summary,
            description="总结长文本，提炼核心知识点。",
        ),
    ]

    agent = create_react_agent(llm, tools)
    logger.info("Built ReAct agent with %d tools", len(tools))
    return agent
