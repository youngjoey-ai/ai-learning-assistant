"""
LangChain 管道构建模块。

提供以下工厂方法：
  - RAG 精准问答
  - 基于知识库的 Agent 任务执行
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import datetime as dt
import operator
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever


AGENT_SYSTEM_PROMPT = """你是一个基于用户已上传知识库执行任务的学习助手。

【工作规则】
1. 所有用户请求都先进行任务判断，并优先调用工具，不要跳过工具直接回答。
2. 只要用户的问题与知识库、文档、学习笔记、项目总结、概念解释、提炼重点、归纳总结有关，优先使用 `knowledge_answer`。
3. 如果需要查看原始片段、逐条摘录或核对来源，再使用 `knowledge_query`。
4. 当用户询问的问题明显超出本地知识库范围（如最新时事、公共常识、知识库中找不到的信息等），再使用 `web_search` 工具在互联网上搜索。
5. 不要在未检索知识库前，就直接回复“请提供内容”“请补充材料”“请说明具体内容”。
6. 如果工具返回“知识库中无相关信息”，再明确告诉用户知识库里没有对应内容，并尝试使用 `web_search` 补充回答。
7. 回答使用中文，尽量直接完成任务；如果基于知识库或搜索结果作答，结尾尽量附上信息来源。"""

TASK_AGENT_SYSTEM_PROMPT = """你是一个任务执行型Agent，目标是完成真实任务而不只是问答。

【工作规则】
1. 优先通过工具完成任务，不要只给建议。
2. 需要实时信息时使用 `web_search`。
3. 需要计算时使用 `calculate`。
4. 需要时间信息时使用 `current_time`。
5. 需要整理长文本时使用 `text_summary`。
6. 如果工具结果不足，再明确说明缺口并给出下一步操作建议。"""

AGENT_THREAD_ID = "study-knowledge-agent-thread"
AGENT_MEMORY = MemorySaver()

RAG_SYSTEM_PROMPT = """\
你是一个严格依据知识库回答问题的助手。

【严格规则】
1. 只允许使用参考文档中的内容回答，禁止编造
2. 如果参考文档中没有相关内容，直接回复：知识库中无相关信息
3. 回答要简洁、清晰、分点说明
4. 回答结尾请附上信息来源"""


@dataclass
class RAGResult:
    """RAG 管道的结构化结果。"""

    answer: str
    source_docs: list[Document]


@dataclass
class AgentRunDetails:
    """Agent 执行的详细结果，包含可观测性指标和轨迹。"""

    answer: str
    metrics: dict[str, Any]
    trace: list[dict[str, Any]]


@dataclass
class AgentToolkit:
    """单 Agent 与工作流 Agent 运行时的工具集合。"""

    knowledge_query: Callable[[str], str]
    knowledge_answer: Callable[[str], str]
    content_summary: Callable[[str], str]
    web_search: Callable[[str], str]


_ALLOWED_OPERATORS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def response_text(response: Any) -> str:
    """从 LangChain 响应中提取可显示的纯文本字符串。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def format_retrieved_docs(docs: list[Document]) -> str:
    """将检索到的文档格式化为包含来源元数据的文本，便于用于提示与展示。"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        formatted.append(f"【片段{idx}】（{source} - 页码{page}）：\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel):
    """构建可调用的 RAG 管道，返回答案与参考文档。"""
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
    """基于检索到的资料生成有依据的回答或总结。"""
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
    """检测需要回退至检索回答的通用 Agent 回复。"""
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


def _build_toolkit(retriever: BaseRetriever, llm: BaseChatModel) -> AgentToolkit:
    """创建依赖同一检索器与 LLM 的工具函数集合。"""

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

    web_search = DuckDuckGoSearchRun()

    def web_search_tool(query: str) -> str:
        try:
            return web_search.run(query)
        except Exception as e:
            return f"搜索失败: {str(e)}"

    return AgentToolkit(
        knowledge_query=knowledge_query,
        knowledge_answer=knowledge_answer,
        content_summary=content_summary,
        web_search=web_search_tool,
    )


def _safe_calculate(expression: str) -> str:
    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return _ALLOWED_OPERATORS[type(node.op)](left, right)
        raise ValueError("仅支持基础四则运算与幂运算")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = eval_node(parsed.body)
        if result.is_integer():
            return str(int(result))
        return str(round(result, 8))
    except Exception as exc:
        return f"计算失败：{exc}"


def build_agent(retriever: BaseRetriever, llm: BaseChatModel):
    """构建具备检索能力的知识库 Agent。"""
    toolkit = _build_toolkit(retriever, llm)

    tools = [
        Tool(
            name="knowledge_answer",
            func=toolkit.knowledge_answer,
            description=(
                "当用户要求基于知识库进行回答、总结、归纳、提炼重点、解释概念、"
                "列出要点时优先使用。输入用户原始任务，工具会自动检索并给出基于知识库的答案。"
            ),
        ),
        Tool(
            name="knowledge_query",
            func=toolkit.knowledge_query,
            description="用于查询知识库原始片段和来源。当需要查看证据、引用原文、核对细节时使用。",
        ),
        Tool(
            name="content_summary",
            func=toolkit.content_summary,
            description="用于总结已有文本内容，提炼核心知识点。通常在获得检索结果后再使用。",
        ),
        Tool(
            name="web_search",
            func=toolkit.web_search,
            description="当知识库中没有相关信息，或者用户询问最新时事、通用常识时，使用此工具在互联网上搜索信息。输入应当是一个简短的搜索关键词。",
        ),
    ]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=AGENT_MEMORY,
    )


def build_task_agent(llm: BaseChatModel):
    web_search = DuckDuckGoSearchRun()

    def search_task_web(query: str) -> str:
        try:
            return web_search.run(query)
        except Exception as exc:
            return f"搜索失败: {exc}"

    def calculate(expression: str) -> str:
        return _safe_calculate(expression)

    def current_time(_: str) -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def text_summary(text: str) -> str:
        response = llm.invoke(
            "请将以下内容整理为可执行的任务清单，按优先级分点：\n"
            f"{text}"
        )
        return response_text(response)

    tools = [
        Tool(
            name="web_search",
            func=search_task_web,
            description="查询实时信息、公开事实、新闻动态、官方文档说明。",
        ),
        Tool(
            name="calculate",
            func=calculate,
            description="执行数学计算。输入示例：'((25+17)*3)/2'。",
        ),
        Tool(
            name="current_time",
            func=current_time,
            description="获取当前本地时间。输入任意文本均可。",
        ),
        Tool(
            name="text_summary",
            func=text_summary,
            description="把长文本整理成结构化任务清单或简明摘要。",
        ),
    ]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=TASK_AGENT_SYSTEM_PROMPT,
        checkpointer=AGENT_MEMORY,
    )


def _collect_tool_trace(result_messages: list[Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for msg in result_messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "tool":
            trace.append({
                "node": "tool_call",
                "tool": getattr(msg, "name", "unknown"),
            })
    return trace


def _invoke_with_timeout(
    agent: Any,
    payload: dict[str, Any],
    config: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.invoke, payload, config)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Agent 执行超时（>{timeout_seconds:.0f}s）") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def run_agent_task(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    messages: list[tuple[str, str]],
    latest_task: str,
    thread_id: str | None = None,
    *,
    return_details: bool = False,
    timeout_seconds: float = 45.0,
) -> str | AgentRunDetails:
    """运行单 Agent 流程，并提供轻量可观测性信息。"""
    resolved_thread_id = thread_id or AGENT_THREAD_ID
    start = time.perf_counter()
    agent = build_agent(retriever, llm)
    result = _invoke_with_timeout(
        agent,
        {"messages": messages},
        {"configurable": {"thread_id": resolved_thread_id}},
        timeout_seconds,
    )
    result_messages = result.get("messages", [])
    ai_messages = [
        msg for msg in result_messages
        if hasattr(msg, "type") and msg.type == "ai"
    ]
    answer = response_text(ai_messages[-1]) if ai_messages else "Agent 未返回有效结果"
    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    trace = _collect_tool_trace(result_messages)
    tool_names = [item["tool"] for item in trace if item.get("tool")]
    metrics: dict[str, Any] = {
        "total_latency_ms": total_latency_ms,
        "tool_call_count": len(tool_names),
        "web_search_used": "web_search" in tool_names,
    }

    if _should_retry_with_grounded_answer(answer):
        docs = retriever.invoke(latest_task)
        answer = _grounded_knowledge_answer(latest_task, docs, llm)
        metrics["fallback_grounded_retry"] = int(metrics.get("fallback_grounded_retry", 0)) + 1

    if return_details:
        return AgentRunDetails(answer=answer, metrics=metrics, trace=trace)
    return answer


def run_task_agent(
    llm: BaseChatModel,
    messages: list[tuple[str, str]],
    latest_task: str,
    thread_id: str | None = None,
    *,
    return_details: bool = False,
    timeout_seconds: float = 45.0,
) -> str | AgentRunDetails:
    resolved_thread_id = thread_id or AGENT_THREAD_ID
    start = time.perf_counter()
    agent = build_task_agent(llm)
    result = _invoke_with_timeout(
        agent,
        {"messages": messages},
        {"configurable": {"thread_id": resolved_thread_id}},
        timeout_seconds,
    )
    result_messages = result.get("messages", [])
    ai_messages = [
        msg for msg in result_messages
        if hasattr(msg, "type") and msg.type == "ai"
    ]
    answer = response_text(ai_messages[-1]) if ai_messages else "Agent 未返回有效结果"
    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    trace = _collect_tool_trace(result_messages)
    tool_names = [item["tool"] for item in trace if item.get("tool")]
    metrics: dict[str, Any] = {
        "task_mode": "executor",
        "total_latency_ms": total_latency_ms,
        "tool_call_count": len(tool_names),
        "web_search_used": "web_search" in tool_names,
        "calculate_used": "calculate" in tool_names,
    }

    if return_details:
        return AgentRunDetails(answer=answer, metrics=metrics, trace=trace)
    return answer
