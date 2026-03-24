# AI Knowledge Assistant

面向个人与团队文档场景的 AI 知识助手，支持文档上传、知识库构建、精准问答、多轮任务处理与会话导出。项目基于 `Streamlit`、`LangChain`、`LangGraph`、`FAISS` 与 `DashScope` 实现，可本地运行，也可直接部署到 `Streamlit Cloud`。

它适合用作以下类型的真实业务工具：

- 内部知识库问答系统
- 培训资料与操作手册助手
- 项目文档检索与总结助手
- 个人资料库与学习资料管理工具

## 核心能力

- 支持上传 `TXT`、`PDF` 文档并自动完成切分、向量化与索引构建
- 内置 `TXT` 编码兜底策略，兼容 `utf-8`、`gb18030`、`gbk`
- 基于 `FAISS` 的本地向量库存储，支持历史知识库自动加载
- 提供 RAG 精准问答模式，回答严格基于检索结果并展示来源
- 提供 Agent 智能助手模式，支持总结、归纳、知识点提炼等复杂任务
- 支持清空知识库、清空对话记录和导出 Markdown 对话记录
- 支持本地 `.env` 配置与 `Streamlit Secrets` 云端配置

## 产品特性

### 1. 文档接入与知识沉淀

上传文档后，系统会自动完成临时落盘、格式识别、文本抽取、分块处理和向量化建库。处理完成后，知识库可持久化保存，应用重启后仍可直接使用。

### 2. 可控的知识库问答

RAG 模式仅基于检索到的文档片段回答问题，并在结果中附带参考来源，适合对准确性、可追溯性要求较高的知识查询场景。

### 3. 面向任务的智能助手

Agent 模式内置知识检索、知识回答和内容总结能力，适合执行“总结文档重点”“整理项目要点”“解释概念差异”“提炼关键信息”等更复杂的任务。

### 4. 面向部署的工程结构

项目已拆分为配置管理、文档处理、链路构建、向量库管理和聊天 UI 辅助等独立模块，便于继续扩展文件格式、替换模型或接入更多工具。

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
├── ai_assistant_app.py       # 主应用入口
├── config.py                 # 配置与密钥解析
├── chains.py                 # RAG / Agent 链路构建
├── document_processing.py    # 文档加载、切分与扩展策略
├── vector_store_manager.py   # 向量库构建、加载、清理与检索接口
├── chat_helpers.py           # 对话渲染、导出与会话控制
├── requirements.txt          # 依赖列表
├── .env.example              # 环境变量示例
├── README.md                 # 项目说明
└── Day1 ~ Day18/             # 历史迭代版本与早期原型参考
```

说明：

- 线上或本地运行请使用 `ai_assistant_app.py`
- 根目录模块为当前主线实现
- `Day1 ~ Day18` 目录属于历史演进代码，不影响当前应用运行

## 系统架构

```text
用户上传 TXT / PDF
        ↓
DocumentProcessor
        ↓
文本切分与清洗
        ↓
DashScope Embeddings
        ↓
FAISS 本地向量库持久化
        ↓
RAG 问答 / Agent 任务执行
        ↓
Streamlit UI 展示与会话导出
```

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd ai_assistant_app
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

如果使用 `uv`：

```bash
uv venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

或：

```bash
uv pip install -r requirements.txt
```

### 4. 配置环境变量

复制示例文件：

```bash
cp .env.example .env
```

在 `.env` 中填写：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

如果部署到 `Streamlit Cloud`，请在 `Secrets` 中配置：

```toml
DASHSCOPE_API_KEY = "your_dashscope_api_key_here"
```

### 5. 启动应用

```bash
streamlit run ai_assistant_app.py
```

默认启动后即可在浏览器中访问本地应用。

## 使用流程

1. 在“知识库上传”页面上传 `TXT` 或 `PDF` 文档
2. 系统自动完成文本切分、向量化与知识库保存
3. 在“RAG 精准问答”页面进行可追溯问答
4. 在“Agent 智能助手”页面执行总结、归纳、提炼等任务
5. 按需导出对话记录或清空当前知识库

## 配置说明

当前默认配置集中在 `config.py`，包括：

- 嵌入模型与聊天模型
- 文本分块大小与重叠长度
- 检索 `Top K`
- 支持的文件类型
- 向量库本地存储路径

默认向量库存储目录：

```text
./saved_vector_store
```

应用启动时会自动尝试加载该目录中的历史知识库。

## 部署说明

项目默认适合以下部署方式：

- 本地单机运行
- 内网演示环境部署
- `Streamlit Cloud` 快速发布

部署时建议注意：

- 配置 `DASHSCOPE_API_KEY`
- 保证运行目录具备本地写权限，用于保存向量库
- 生产环境可通过挂载持久化卷保留知识库数据
- 如需团队协作场景，可进一步增加鉴权、用户隔离和对象存储能力

## 典型应用场景

- 企业制度、SOP、培训手册问答
- 项目方案、周报、复盘文档整理
- 招投标材料、产品资料、需求文档检索
- 个人知识库、研究资料和课程资料问答

## 后续扩展方向

- 增加 `DOCX`、`Markdown` 等更多文档格式支持
- 接入多知识库管理与用户权限控制
- 增加引用高亮、检索调参和管理后台
- 替换或并行接入更多大模型与向量数据库
