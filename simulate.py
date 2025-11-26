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
from pr_strategies.strategies import (
    DoNothingStrategy,
    DelayedApologyStrategy,
    FastClarifyStrategy,
)
from utils.spread_model import HawkesProcess


def build_agents(llm: LLMClient) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}

    agents["BrandOfficial"] = Agent(
        name="BrandOfficial",
        role="brand_official",
        profile="星光电子官方账号，目标是维护品牌形象，但需要承担责任。",
        llm_client=llm,
    )

    for i in range(5):
        name = f"AngryUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="angry_user",
            profile="对数据泄露非常愤怒，容易发表激烈批评。",
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
            profile="长期使用星光电子产品，倾向于为品牌辩护。",
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


def inject_initial_rumor(env: SocialEnv):
    """
    t=0，媒体发第一条负面爆料。
    """
    env.t = 0
    any_agent = next(iter(env.agents.values()))
    llm = any_agent.llm

    system = "你是一个科技媒体账号，语气略带煽动。"
    user = """写一条关于“星光电子用户数据疑似大规模泄露”的爆料帖，
可以质疑其安全性，但不要太长，1~2 句话。"""
    text = llm.chat(system, user)

    env._add_post(
        author="Media1",
        text=text,
        sentiment="NEGATIVE",
        tag="rumor",
        target_post_id=None,
    )


def create_simulation_instance(strategy_name: str, seed: int = 42):
    """
    创建完整模拟实例：llm + agents + graph + env + pr_strategy。
    不同策略共用同一随机种子，方便对比。
    """
    random.seed(seed)

    llm = LLMClient()
    agents = build_agents(llm)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G)

    if strategy_name == "S0":
        strategy = DoNothingStrategy(brand_name="BrandOfficial")
    elif strategy_name == "S1":
        strategy = DelayedApologyStrategy(brand_name="BrandOfficial")
    elif strategy_name == "S2":
        strategy = FastClarifyStrategy(brand_name="BrandOfficial")
    else:
        raise ValueError(f"未知策略: {strategy_name}")

    inject_initial_rumor(env)

    return env, strategy


def run_once(strategy_name: str = "S0", T: int = 10, seed: int = 42):
    random.seed(seed)

    llm = LLMClient()
    agents = build_agents(llm)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G)

    if strategy_name == "S0":
        strategy = DoNothingStrategy(brand_name="BrandOfficial")
    elif strategy_name == "S1":
        strategy = DelayedApologyStrategy(brand_name="BrandOfficial")
    elif strategy_name == "S2":
        strategy = FastClarifyStrategy(brand_name="BrandOfficial")
    else:
        raise ValueError("未知策略")

    inject_initial_rumor(env)

    for t in range(1, T + 1):
        print(f"=== 时间步 {t} ===")
        new_posts = env.step(pr_strategy=strategy)
        for p in new_posts:
            print(f"[{p.time_step}] {p.author}: {p.text} (sentiment={p.sentiment})")

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
    df = run_once(strategy_name="S1", T=8, seed=123)
    print("总帖子数:", len(df))

    # 话题热度 + 霍克斯过程 + 多代理容器演示（简版）
    topics = ["数据泄露", "品牌回应", "用户情绪"]
    llm = LLMClient()

    def make_topic_agent(idx, tps):
        return Agent(
            name=f"TopicAgent{idx}",
            role="generic",
            profile="关注多话题的用户",
            llm_client=llm,
            topics=list(tps),
        )

    hawkes_params = {"alpha": 1.0, "beta": 0.5, "mu": 0.1}
    topic_manager = TopicManager(topics, hawkes_params=hawkes_params)
    hawkes_process = HawkesProcess(**hawkes_params)
    system = MultiAgentSystem(
        agent_count=5,
        topics=topics,
        agent_factory=make_topic_agent,
    )

    for step in range(10):
        system.run_simulation_step(topic_manager)
        if hawkes_process.simulate_event(step):
            topic_sel = np.random.choice(topics)
            topic_manager.add_post(topic_sel, f"外部事件触发（t={step}）", current_time=step)
        heats = {t: topic_manager.get_heat(t) for t in topics}
        print(f"[Topic demo] t={step}, heats={heats}")
