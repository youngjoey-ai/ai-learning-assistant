"""
LangChain pipeline construction module.

Provides factory helpers for:
  - RAG question answering
  - Knowledge-grounded agent task execution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever


AGENT_SYSTEM_PROMPT = """你是一个基于用户已上传知识库执行任务的学习助手。

【工作规则】
1. 只要用户的问题与知识库、文档、学习笔记、项目总结、概念解释、提炼重点、归纳总结有关，优先使用 `knowledge_answer`。
2. 如果需要查看原始片段、逐条摘录或核对来源，再使用 `knowledge_query`。
3. 不要在未检索知识库前，就直接回复“请提供内容”“请补充材料”“请说明具体内容”。
4. 如果工具返回“知识库中无相关信息”，再明确告诉用户知识库里没有对应内容。
5. 回答使用中文，尽量直接完成任务；如果基于知识库作答，结尾尽量附上信息来源。"""

RAG_SYSTEM_PROMPT = """\
你是一个严格依据知识库回答问题的助手。

【严格规则】
1. 只允许使用参考文档中的内容回答，禁止编造
2. 如果参考文档中没有相关内容，直接回复：知识库中无相关信息
3. 回答要简洁、清晰、分点说明
4. 回答结尾请附上信息来源"""


@dataclass
class RAGResult:
    """Structured result from the RAG pipeline."""

    answer: str
    source_docs: list[Document]


def response_text(response: Any) -> str:
    """Extract a plain displayable string from LangChain responses."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def format_retrieved_docs(docs: list[Document]) -> str:
    """Format documents with source metadata for prompts and display."""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        formatted.append(f"【片段{idx}】（{source} - 页码{page}）：\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel):
    """Build a callable RAG pipeline that returns answer + source docs."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "【参考文档】\n{context}\n\n【用户问题】\n{question}"),
    ])

    def invoke(question: str) -> RAGResult:
        source_docs = retriever.invoke(question)
        response = (prompt | llm).invoke({
            "context": format_retrieved_docs(source_docs),
            "question": question,
        })
        return RAGResult(
            answer=response_text(response),
            source_docs=source_docs,
        )

    return invoke


def _grounded_knowledge_answer(
    task: str,
    docs: list[Document],
    llm: BaseChatModel,
) -> str:
    """Generate a grounded answer or summary from retrieved knowledge docs."""
    if not docs:
        return "知识库中无相关信息"

    grounded_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格依据知识库完成任务的学习助手。

【严格规则】
1. 只允许依据参考资料完成任务，禁止编造。
2. 如果参考资料不足以支持回答，直接回复：知识库中无相关信息。
3. 按照用户任务要求输出，可以总结、归纳、提炼重点、解释概念或回答问题。
4. 回答简洁清晰、优先分点。
5. 回答结尾附上信息来源。"""),
        ("human", "【用户任务】\n{task}\n\n【参考资料】\n{context}"),
    ])

    response = (grounded_prompt | llm).invoke({
        "task": task,
        "context": format_retrieved_docs(docs),
    })
    return response_text(response)


def _should_retry_with_grounded_answer(answer: str) -> bool:
    """Detect generic agent replies that should be retried with retrieval."""
    retry_markers = (
        "请提供",
        "请先提供",
        "请补充",
        "需要具体",
        "请说明具体",
        "未提供",
        "需要更多上下文",
        "无法根据现有信息",
    )
    return (
        "知识库中无相关信息" not in answer
        and any(marker in answer for marker in retry_markers)
    )


def build_agent(retriever: BaseRetriever, llm: BaseChatModel):
    """Build a knowledge-grounded agent with retrieval-aware tools."""

    def knowledge_query(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中无相关信息"
        return format_retrieved_docs(docs)

    def knowledge_answer(task: str) -> str:
        docs = retriever.invoke(task)
        return _grounded_knowledge_answer(task, docs, llm)

    def content_summary(text: str) -> str:
        response = llm.invoke(
            "请基于输入内容进行简洁总结，突出核心知识点，必要时分点回答：\n"
            f"{text}"
        )
        return response_text(response)

    tools = [
        Tool(
            name="knowledge_answer",
            func=knowledge_answer,
            description=(
                "当用户要求基于知识库进行回答、总结、归纳、提炼重点、解释概念、"
                "列出要点时优先使用。输入用户原始任务，工具会自动检索并给出基于知识库的答案。"
            ),
        ),
        Tool(
            name="knowledge_query",
            func=knowledge_query,
            description="用于查询知识库原始片段和来源。当需要查看证据、引用原文、核对细节时使用。",
        ),
        Tool(
            name="content_summary",
            func=content_summary,
            description="用于总结已有文本内容，提炼核心知识点。通常在获得检索结果后再使用。",
        ),
    ]

    return create_agent(
        llm,
        tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        name="study_knowledge_agent",
    )


def run_agent_task(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    messages: list[tuple[str, str]],
    latest_task: str,
) -> str:
    """Run the agent with conversation history and a grounded fallback."""
    agent = build_agent(retriever, llm)
    result = agent.invoke({"messages": messages})
    ai_messages = [
        msg for msg in result.get("messages", [])
        if hasattr(msg, "type") and msg.type == "ai"
    ]

    if ai_messages:
        answer = response_text(ai_messages[-1])
    else:
        answer = "Agent 未返回有效结果"

    if _should_retry_with_grounded_answer(answer):
        docs = retriever.invoke(latest_task)
        return _grounded_knowledge_answer(latest_task, docs, llm)

    return answer
