# AI Learning Assistant

一个可以独立运行、独立部署的 AI 学习助手项目。

仓库根目录放的是最终成品版本，可以直接启动体验；`Day1` 到 `Day18` 记录了这个项目从 Python 基础、文本处理、LangChain、RAG 到 Agent 的逐步学习和完善过程。也就是说，这个仓库既是一个完整项目，也是它从 0 到 1 的成长轨迹。

## 项目定位

- 根目录是最终可运行版本，相当于 `Day19` 的独立项目成果
- `Day1` 到 `Day18` 不是零散笔记，而是最终项目能力逐步搭建的过程
- 如果你只想体验成品，直接看根目录
- 如果你想看这个项目是怎么一步步做出来的，可以按 `Day1` 到 `Day18` 顺序阅读

## 最终项目能做什么

根目录的应用基于 `Streamlit + LangChain + LangGraph + FAISS + DashScope`，目前支持：

- 上传 `TXT` 和 `PDF` 文档构建个人知识库
- 自动切分文本并生成向量索引
- 基于 RAG 的知识库精准问答
- 在回答中展示参考来源，减少幻觉
- 基于 Agent 的复杂任务处理，如检索、总结、归纳
- 对话历史清空与导出
- 同时兼容本地 `.env` 和 Streamlit Cloud `Secrets`

## 学习演进路线

这个仓库不是一上来就完成的，而是按天持续迭代出来的。

- `Day1` - `Day3`：Python 基础语法、列表、字典、函数
- `Day4` - `Day7`：文件读写、JSON、类封装、简单学习助手工具
- `Day8` - `Day10`：文本处理、摘要、改写、信息提取
- `Day11` - `Day13`：LangChain 入门、Prompt、Embedding、FAISS 检索
- `Day14` - `Day15`：RAG 核心链路搭建与优化
- `Day16` - `Day17`：Agent、LangGraph、多步骤任务处理
- `Day18`：AI 学习助手应用雏形
- 根目录：最终独立项目版本，可直接运行和部署，视作 `Day19`

## 仓库结构

```text
ai-learning-assistant/
├── ai_assistant_app.py   # 最终版应用主程序（Day19 成果）
├── requirements.txt      # 最终版依赖
├── README.md             # 项目说明
├── Day1/ ~ Day3/         # Python 基础
├── Day4/ ~ Day10/        # 文件处理、文本处理、小工具
├── Day11/ ~ Day13/       # LangChain、Embedding、向量检索
├── Day14/ ~ Day15/       # RAG 实践与优化
├── Day16/ ~ Day17/       # Agent / LangGraph
└── Day18/                # 最终应用雏形
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/youngjoey-ai/ai-learning-assistant.git
cd ai-learning-assistant
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

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

如果部署到 Streamlit Cloud，则在 `Secrets` 中配置：

```toml
DASHSCOPE_API_KEY = "your_api_key_here"
```

### 5. 运行最终项目

```bash
streamlit run ai_assistant_app.py
```

## 阅读建议

- 想看最终成果：先看根目录的 `ai_assistant_app.py`
- 想看学习过程：按 `Day1` 到 `Day18` 顺序阅读
- 想看 RAG 演进：重点看 `Day13`、`Day14`、`Day15`
- 想看 Agent 演进：重点看 `Day16`、`Day17`、`Day18`

## 说明

这个仓库的重点不只是“学了什么”，更是“怎样把每天学到的内容逐步汇总成一个可运行的独立项目”。

所以它既适合：

- 当作 AI 应用学习路径参考
- 当作个人项目成长记录
- 当作一个可以继续扩展的 RAG + Agent 小项目起点
