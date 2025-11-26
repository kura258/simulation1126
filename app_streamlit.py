import time
from typing import List

import pandas as pd
import streamlit as st

from simulate import simulate_steps
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


def main():
    st.set_page_config(page_title="话题热度模拟", layout="wide")
    st.title("多智能体舆论模拟：话题与热度可视化")

    # 控制面板
    st.sidebar.header("模拟参数")
    T = st.sidebar.slider("模拟时间步数", min_value=5, max_value=30, value=10, step=1)
    base_seed = st.sidebar.number_input("随机种子", min_value=0, max_value=9999, value=42)
    delay_sec = st.sidebar.slider("每步界面延迟（秒）", 0.0, 2.0, 0.2, 0.05)
    request_delay = st.sidebar.slider("API 请求间隔（秒）", 0.0, 2.0, 0.2, 0.05)
    topics_input = st.sidebar.text_area(
        "自定义话题（逗号或换行分隔）",
        "数据安全,品牌回应,用户情绪",
        height=80,
        placeholder="示例：\n数据安全\n品牌回应\n用户情绪",
    )
    raw_topics = topics_input.replace("\n", ",")
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]

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


if __name__ == "__main__":
    main()
