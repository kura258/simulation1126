# simulate.py
from __future__ import annotations
import os
import random
import sys
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

# 把项目根目录加入模块搜索路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agents.agent import Agent
from agents.llm_client import LLMClient
from agents.multi_agent_system import MultiAgentSystem
from env.social_env import SocialEnv, TopicManager
from utils.spread_model import HawkesProcess


def build_agents(llm: LLMClient) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}

    agents["BrandOfficial"] = Agent(
        name="BrandOfficial",
        role="brand_official",
        profile="品牌官方账号，目标是维护品牌形象并稳定舆论。",
        llm_client=llm,
    )

    for i in range(5):
        name = f"AngryUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="angry_user",
            profile="对负面事件非常愤怒，容易发表激烈批评。",
            llm_client=llm,
        )

    for i in range(5):
        name = f"NeutralUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="neutral_user",
            profile="对事件保持观望，容易被他人观点影响。",
            llm_client=llm,
        )

    for i in range(3):
        name = f"FanUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="fan_user",
            profile="长期关注该品牌，倾向于为品牌辩护。",
            llm_client=llm,
        )

    agents["Media1"] = Agent(
        name="Media1",
        role="media",
        profile="科技媒体账号，追求流量，也关心事实。",
        llm_client=llm,
    )

    return agents


def build_graph(agent_names):
    G = nx.DiGraph()
    G.add_nodes_from(agent_names)

    # 所有人都关注官方账号与媒体
    for name in agent_names:
        if name != "BrandOfficial":
            G.add_edge(name, "BrandOfficial")
        if name != "Media1":
            G.add_edge(name, "Media1")

    # 再随机补充一些关注关系
    names = list(agent_names)
    for src in names:
        for _ in range(3):
            dst = random.choice(names)
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst)

    return G


def inject_initial_rumor(env: SocialEnv, topic: str | None = None):
    """
    t=0，媒体发第一条热点爆料。
    """
    env.t = 0
    any_agent = next(iter(env.agents.values()))
    llm = any_agent.llm

    system = "你是一个科技媒体账号，语气略带煽动。"
    user = """写一条关于“热点科技产品存在安全隐患”的爆料帖，
可以质疑其可靠性，但不要太长，1~2 句话。"""
    text = llm.chat(system, user)

    env._add_post(
        author="Media1",
        text=text,
        sentiment="NEGATIVE",
        tag="rumor",
        target_post_id=None,
        topic=topic,
    )


def simulate_steps(
    T: int = 10,
    seed: int = 42,
    topics: list[str] | None = None,
    request_delay: float = 0.0,
):
    """
    运行多时间步模拟，返回环境、每步新增帖子列表、以及话题热度快照。
    """
    random.seed(seed)

    llm = LLMClient()
    agents = build_agents(llm)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G, topics=topics)

    # 初始爆料关联首个话题
    topic0 = topics[0] if topics else None
    inject_initial_rumor(env, topic=topic0)

    steps = []
    heat_history = []
    for t in range(1, T + 1):
        new_posts = env.step(request_delay=request_delay)
        steps.append(new_posts)
        if env.topic_manager:
            snapshot = {"time": env.t}
            for topic in env.topic_manager.topics:
                snapshot[topic] = env.topic_manager.get_heat(topic)
            heat_history.append(snapshot)
    return env, steps, heat_history


def run_once(T: int = 10, seed: int = 42):
    env, steps, _ = simulate_steps(T=T, seed=seed)

    rows = []
    for p in env.posts:
        rows.append({
            "time": p.time_step,
            "author": p.author,
            "sentiment": p.sentiment,
            "tag": p.tag,
            "text": p.text,
        })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = run_once(T=8, seed=123)
    print("总帖子数:", len(df))
