import time
from typing import List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from simulate import simulate_steps, pick_default_topics
from env.social_env import SocialEnv


def collect_heat_history_df(heat_history: List[dict]) -> pd.DataFrame:
    if not heat_history:
        return pd.DataFrame(columns=["time"])
    return pd.DataFrame(heat_history)


def collect_posts_df(env: SocialEnv) -> pd.DataFrame:
    rows = []
    for p in env.posts:
        rows.append({
            "time": p.time_step,
            "author": p.author,
            "sentiment": p.sentiment,
            "topic": p.topic or "未标注",
            "tag": p.tag,
            "text": p.text,
        })
    return pd.DataFrame(rows)


def collect_agent_timeline(steps: List[List], agents) -> pd.DataFrame:
    """
    汇总每个 Agent 在每个时间步的行为（post/retweet/silent），并附带情绪与话题。
    """
    rows = []
    agent_names = list(agents.keys())
    for t_idx, posts in enumerate(steps, start=1):
        for name in agent_names:
            agent_posts = [p for p in posts if p.author == name]
            if agent_posts:
                for p in agent_posts:
                    rows.append({
                        "time": t_idx,
                        "agent": name,
                        "action": "post" if p.tag != "retweet" else "retweet",
                        "sentiment": p.sentiment,
                        "topic": p.topic or "未标注",
                        "text": p.text,
                    })
            else:
                rows.append({
                    "time": t_idx,
                    "agent": name,
                    "action": "silent",
                    "sentiment": "NEUTRAL",
                    "topic": "无",
                    "text": "",
                })
    return pd.DataFrame(rows)


def compute_pred_df(heat_history: List[dict]) -> pd.DataFrame:
    if not heat_history:
        return pd.DataFrame(columns=["time", "topic", "heat_pred"])
    wide = pd.DataFrame(heat_history)
    long = wide.melt(id_vars=["time"], var_name="topic", value_name="heat_pred")
    return long


def compute_metrics(pred_df: pd.DataFrame, real_df: pd.DataFrame):
    merged = pred_df.merge(real_df, on=["time", "topic"], how="inner")
    if merged.empty:
        return None, None, None
    merged["mse"] = (merged["heat_pred"] - merged["heat_real"]) ** 2
    merged["ape"] = (merged["heat_pred"] - merged["heat_real"]).abs() / (merged["heat_real"].abs() + 1e-6)
    overall_mse = float(merged["mse"].mean())
    overall_mape = float(merged["ape"].mean() * 100)
    per_topic = merged.groupby("topic").agg(
        mape=("ape", lambda x: float(x.mean() * 100)),
        mse=("mse", "mean"),
    ).reset_index()
    return overall_mse, overall_mape, per_topic


def plot_metrics(per_topic: pd.DataFrame):
    if per_topic is None or per_topic.empty:
        return None
    per_topic = per_topic.sort_values("mape", ascending=False)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(per_topic["topic"], per_topic["mape"], color="#4C72B0")
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title("Per-topic MAPE")
    axes[0].invert_yaxis()

    axes[1].barh(per_topic["topic"], per_topic["mse"], color="#55A868")
    axes[1].set_xlabel("MSE")
    axes[1].set_title("Per-topic MSE")
    axes[1].invert_yaxis()

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="话题热度模拟", layout="wide")
    st.title("多智能体舆论模拟：话题与热度可视化")

    # 控制面板
    st.sidebar.header("模拟参数")
    T = st.sidebar.slider("模拟时间步数", min_value=10, max_value=120, value=100, step=5)
    base_seed = st.sidebar.number_input("随机种子", min_value=0, max_value=9999, value=42)
    delay_sec = st.sidebar.slider("每步界面延迟（秒）", 0.0, 2.0, 0.2, 0.05)
    request_delay = st.sidebar.slider("API 请求间隔（秒）", 0.0, 2.0, 0.2, 0.05)

    # 默认话题：从 classified_events_35.csv 抽取 5 个，可在前端修改
    if "topics_input" not in st.session_state:
        default_topics = pick_default_topics(seed=base_seed, k=5) or ["数据安全", "全运会夺冠", "明星结婚"]
        st.session_state["topics_input"] = "\n".join(default_topics)

    topics_input = st.sidebar.text_area(
        "自定义话题（逗号或换行分隔）",
        st.session_state["topics_input"],
        height=80,
        placeholder="示例：\n数据安全\n全运会夺冠\n明星结婚",
    )
    st.session_state["topics_input"] = topics_input
    raw_topics = topics_input.replace("\n", ",")
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    real_file = st.sidebar.file_uploader("上传真实热度 CSV (列: time, topic, heat)", type=["csv"])

    if st.button("开始模拟"):
        st.info("正在创建环境并运行，请稍候...")
        env, steps, heat_history = simulate_steps(
            T=T,
            seed=base_seed,
            topics=topics,
            request_delay=request_delay,
        )
        st.success("模拟完成")

        # 话题热度折线图
        if heat_history:
            heat_df = collect_heat_history_df(heat_history).set_index("time")
            st.subheader("话题热度随时间变化")
            st.line_chart(heat_df)
        else:
            st.info("当前未配置话题或无热度数据。")

        # 时间步帖子展示
        st.subheader("按时间步的事件内容")
        for t_idx, posts in enumerate(steps, start=1):
            with st.expander(f"时间步 {t_idx} ({len(posts)} 条)"):
                if posts:
                    rows = [{
                        "作者": p.author,
                        "情绪": p.sentiment,
                        "话题": p.topic or "未标注",
                        "标签": p.tag,
                        "内容": p.text,
                    } for p in posts]
                    st.table(pd.DataFrame(rows))
                else:
                    st.write("无新帖子")
                if delay_sec > 0 and t_idx < len(steps):
                    time.sleep(delay_sec)

        # Agent 行为时间线
        st.subheader("Agent 行为时间线")
        agent_timeline = collect_agent_timeline(steps, env.agents)
        st.dataframe(agent_timeline)

        # 汇总所有帖子表
        st.subheader("全部帖子汇总")
        posts_df = collect_posts_df(env)
        st.dataframe(posts_df)

        # 真实数据对比
        if real_file:
            try:
                real_df = pd.read_csv(real_file)
                if {"time", "topic", "heat"} <= set(real_df.columns):
                    real_df = real_df.rename(columns={"heat": "heat_real"})
                    pred_df = compute_pred_df(heat_history)
                    overall_mse, overall_mape, per_topic = compute_metrics(pred_df, real_df)
                    st.subheader("真实数据对比 (MAPE / MSE)")
                    if overall_mse is not None:
                        st.write(f"总体 MAPE: {overall_mape:.3f}%")
                        st.write(f"总体 MSE: {overall_mse:.6f}")
                        st.dataframe(per_topic)
                        fig = plot_metrics(per_topic)
                        if fig:
                            st.pyplot(fig)
                    else:
                        st.info("无法对齐真实数据，请确认 time/topic 匹配。")
                else:
                    st.warning("真实数据需包含列: time, topic, heat")
            except Exception as e:
                st.error(f"真实数据处理失败: {e}")


if __name__ == "__main__":
    main()
