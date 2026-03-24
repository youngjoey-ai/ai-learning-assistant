# AI Learning Assistant

基于 Streamlit、LangChain、LangGraph 和 FAISS 的 AI 学习助手，支持上传 `TXT` / `PDF` 文档，构建本地知识库，并提供 RAG 精准问答与 Agent 智能助手两种使用模式。

## 功能亮点

- 支持上传 `TXT`、`PDF` 文档并自动切分文本
- 对 `TXT` 文件做编码兜底，兼容 `utf-8`、`gb18030`、`gbk`
- 使用 DashScope Embeddings 构建 FAISS 向量库
- 启动时自动加载历史向量库，减少重复处理
- 提供 RAG 精准问答，回答时附带参考来源
- 提供 Agent 智能助手，支持检索、总结、归纳等复杂任务
- 支持清空和导出对话记录
- 兼容本地 `.env` 和 Streamlit Cloud `Secrets`

## 技术栈

- `Streamlit`
- `LangChain`
- `LangGraph`
- `FAISS`
- `DashScope Embeddings`
- `Qwen Turbo`（通过 OpenAI 兼容接口调用）

## 项目结构

```text
ai_assistant_app/
├── ai_assistant_app.py       # Streamlit 应用主入口
├── config.py                 # 配置定义
├── chains.py                 # RAG / Agent 链路封装
├── document_processing.py    # 文档加载与分块处理
├── vector_store_manager.py   # 向量库管理
├── chat_helpers.py           # 对话 UI 辅助函数
├── requirements.txt          # 依赖列表
├── .env.example              # 环境变量示例
├── README.md                 # 项目说明
└── Day1 ~ Day18/             # 学习过程中的阶段性代码与实验文件
```

说明：

- 当前应用入口是 `ai_assistant_app.py`
- `config.py`、`chains.py`、`document_processing.py` 等文件是后续模块化拆分的产物
- `Day1` 到 `Day18` 目录保留了学习过程中的每日练习与演进版本

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd ai_assistant_app
```

### 2. 创建并激活虚拟环境

如果目录里已经有可用的 `.venv`，可以直接激活；否则先创建：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

如果你使用 `uv`，也可以这样创建环境：

```bash
uv venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果你使用 `uv`：

```bash
uv pip install -r requirements.txt
```

### 4. 配置环境变量

可以直接复制示例文件：

```bash
cp .env.example .env
```

然后在项目根目录的 `.env` 中填写：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

如果部署到 Streamlit Cloud，请在 `Secrets` 中配置同名变量：

```toml
DASHSCOPE_API_KEY = "your_dashscope_api_key_here"
```

### 5. 启动应用

```bash
streamlit run ai_assistant_app.py
```

启动后在浏览器打开本地地址即可使用。

## 使用流程

1. 在“知识库上传”页上传 `TXT` 或 `PDF` 文件
2. 系统自动切分文档并构建向量库
3. 切换到“RAG 精准问答”进行基于知识库的提问
4. 切换到“Agent 智能助手”完成总结、归纳、检索等复杂任务
5. 如有需要，可导出问答记录为 Markdown 文件

## 应用说明

### RAG 精准问答

- 基于知识库检索结果生成回答
- 回答时展示参考文档来源
- 尽量减少幻觉，严格围绕上传内容作答

### Agent 智能助手

- 内置知识库查询工具
- 内置文本总结工具
- 适合执行多步骤学习任务，比如总结重点、归纳概念、提炼知识点

## 向量库持久化

应用会将向量库保存到本地目录：

```text
./saved_vector_store
```

下次启动时会尝试自动加载已保存的知识库。该目录通常在首次上传文档后自动生成。

## 依赖说明

核心依赖见 [requirements.txt](requirements.txt)，包括：

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-openai`
- `langchain-text-splitters`
- `langgraph`
- `dashscope`
- `pypdf`
- `faiss-cpu`

## 部署说明

部署到 Streamlit Cloud 时，请注意：

- 在 `Secrets` 中配置 `DASHSCOPE_API_KEY`
- 确保 [requirements.txt](requirements.txt) 中的依赖完整可安装
- 应用运行目录需要具备本地写入权限，以保存向量库

## 适用场景

- 课程资料问答
- 学习笔记检索
- PDF / TXT 文档总结
- 复习资料整理
- 个人知识库问答助手
