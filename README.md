# 多智能体舆论博弈模拟

一个基于大模型的多智能体舆论模拟小实验，支持话题热度、霍克斯传播、自定义 PR 策略，以及 Streamlit 前端展示。

## 功能亮点
- 预置 3 种 PR 策略：`S0`（不作为）、`S1`（延迟致歉）、`S2`（快速澄清），可扩展自定义策略。
- 多角色 Agent（官方、愤怒用户、中立吃瓜、忠实粉丝、媒体），内置记忆流与反思能力。
- 话题管理与热度：支持多话题注意力分配；每个话题使用霍克斯过程建模自激传播。
- Streamlit 前端：查看时间步帖子列表、情绪标签、策略效果对比。
- 本地小模型嵌入：默认使用 `all-MiniLM-L6-v2`（存放于 `models/`，可自行下载或运行导出脚本）。

## 目录结构
- `simulate.py`：命令行入口，单次运行或策略对比。
- `app_streamlit.py`：前端入口，交互式选择策略/参数。
- `agents/agent.py`：Agent 逻辑，含话题注意力、互动与记忆更新。
- `agents/llm_client.py`：封装 SiliconFlow DeepSeek 接口。
- `agents/memory.py`：记忆流与反思（基于 SentenceTransformer）。
- `agents/multi_agent_system.py`：多代理容器，便于批量驱动互动。
- `env/social_env.py`：社交环境、话题管理器、霍克斯传播。
- `utils/spread_model.py`：霍克斯过程实现。
- `pr_strategies/strategies.py`：PR 策略集合。
- `run_simulation.cmd`：Windows 一键脚本（创建 venv + 运行 Streamlit）。
- `export_embedding_model.py`：示例脚本，下载/导出 `all-MiniLM-L6-v2` 至 `models/`。

## 环境准备
1) 安装 Python 3.10+（推荐 3.11/3.12）。  
2) 创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   # PowerShell: .\venv\Scripts\Activate.ps1
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
3) 配置大模型 API（SiliconFlow DeepSeek-Terminus）  
   在项目根目录创建 `.env`：
   ```env
   SILICONFLOW_API_KEY=你的APIKey
   SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
   SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
   ```
4) 嵌入模型  
   - 推荐在本地下载：运行 `python export_embedding_model.py`，会将 `all-MiniLM-L6-v2` 保存到 `models/all-MiniLM-L6-v2`。  
   - 若不想提交模型文件，请在 `.gitignore` 中忽略 `models/`。

## 运行方式
### 命令行单次模拟
```bash
python simulate.py           # 默认 S1, T=8, seed=123
```
如需修改策略/步数/seed，可调整 `run_once` 调用或编写自己的入口。

### Streamlit 前端
```bash
streamlit run app_streamlit.py
# 浏览器访问 http://localhost:8501 交互体验
```
Windows 可直接运行 `run_simulation.cmd`，自动创建/激活 venv 并启动前端。

### 话题热度与传播演示
`simulate.py` 末尾提供示例：多代理容器 + TopicManager + 霍克斯过程，展示多话题热度随时间的变化。

## 注意事项
- 需要外部 LLM API；无密钥或离线环境会导致示例运行失败，请提前配置 `.env`。
- 模型目录 `models/` 体积较大（约 90MB），如需推送到 GitHub，建议使用 Git LFS 或在 `.gitignore` 中忽略。
- 虚拟环境目录 `.venv/` 不应提交，请在 `.gitignore` 中忽略（当前默认忽略 `venv/`，可追加 `.venv/`）。
- 本项目仅为演示/教学用途，示例数据非真实业务数据，可根据需要调整角色设定和策略规则。
