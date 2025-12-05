# 多智能体舆论博弈模拟（话题热度预测版）

基于多代理 + Hawkes 自激模型的舆论演化小实验，提供训练/评估脚本与 Streamlit 前端。当前默认使用双衰减核（爆发+长尾）拟合 35 个事件数据，并在模拟中内置最优参数。

## 功能概览
- 多角色 Agent：品牌官方、愤怒用户、中立吃瓜、粉丝、媒体；支持话题注意力分配与记忆流。
- 话题热度：TopicManager 记录帖子与热度；传播侧可用 Hawkes 自激；训练侧使用双衰减核（mu_fast/mu_slow/H_base/lambda_fast/lambda_slow）。
- 前端：可输入自定义话题/时间步（默认 100），显示热度曲线、帖子列表、Agent 行为时间线；支持上传真实数据 CSV，对比 MAPE/MSE，并给出按话题的对比图。
- 默认话题：从 `classified_events_35.csv` 的前 5 个唯一 topic 填充侧边栏（仍可自行修改）。
- 可视化：`plots_huoju_norm` 提供 35 个事件的真实/预测曲线；`plots_huoju_norm_overview.png` 为总览。

## 数据
- 训练/评估：`datasets/`（示例 34+ 事件，时间序列 heat）。
- 归一化拆分：`classified_events_35.csv`（topic, heat, timestamp），已按话题拆为 `datasets_huoju*`。
- 默认模拟话题来源：`classified_events_35.csv`（前 5 个 topic）。

## LLM 配置
- 客户端：`agents/llm_client.py` 默认使用 `https://api.openai-proxy.org/v1`，模型 `gpt-4o-mini`，API Key 可在环境变量 `SILICONFLOW_API_KEY` 或代码默认值设置。
- 如需自定义：在 `.env` 中配置 `SILICONFLOW_API_KEY`、`SILICONFLOW_BASE_URL`、`SILICONFLOW_MODEL`。

## 依赖与环境
```bash
python -m venv venv
# PowerShell: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 训练与评估
- 全局参数拟合（双衰减核）：
```bash
python train.py --data_dir datasets_huoju_norm        # 归一化数据
python train.py --data_dir datasets_huoju_norm --random_test --seed 123  # 随机抽 10% 作为 test
```
- 评估脚本：`test_oos.py`（可自填最优参数，输出 MSE/MAPE + 可视化）。

模型形式：
```
pred_t = H_base + mu_fast * M_fast + mu_slow * M_slow
M_fast = M_fast * exp(-lambda_fast) + y_{t-1}
M_slow = M_slow * exp(-lambda_slow) + y_{t-1}
约束：lambda_fast > lambda_slow > 0，所有参数为正
```

## 模拟与前端
- 命令行单次模拟（默认 T=100，话题取前 5 个）：
```bash
python simulate.py
```
- Streamlit 前端：
```bash
streamlit run app_streamlit.py
# 浏览器访问 http://localhost:8501
```
侧边栏可设置时间步数、随机种子、API 请求间隔；显示默认话题（来自 `classified_events_35.csv` 前 5 个），支持自定义输入。上传真实热度 CSV（列：time, topic, heat）后，会计算总体/分话题 MAPE、MSE 并绘制对比图。

## 主要文件
- `simulate.py`：默认最优参数、话题抽取、时间步 100 的多 Agent 模拟。
- `app_streamlit.py`：交互式前端，包含真实数据对比与话题提示。
- `agents/agent.py` / `agents/memory.py` / `agents/llm_client.py`：Agent 逻辑、记忆流、LLM 客户端。
- `env/social_env.py`：社交环境与话题热度管理（兼容 Hawkes 近似参数）。
- `utils/spread_model.py`：双衰减核自激预测器。
- `train.py` / `utils/data_loader.py`：训练数据加载与参数拟合。
- `plots_huoju_norm/*.png`：35 个事件的真实/预测可视化；`plots_huoju_norm_overview.png` 总览。

## 注意
- 需可用的 LLM API Key；如在受限网络，需自行配置代理或离线策略。
- 大模型调用有重试兜底；仍建议检查网络/证书/限流。
- 数据与参数均为示例用途，可替换为自有话题与序列。***
