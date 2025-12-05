# simulate.py
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from pathlib import Path

# 把项目根目录加入模块搜索路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agents.agent import Agent
from agents.llm_client import LLMClient
from env.social_env import SocialEnv
from pr_strategies.strategies import (
    DoNothingStrategy,
    DelayedApologyStrategy,
    FastClarifyStrategy,
)

# 非随机全量训练得到的最优参数（归一化数据）
BEST_HAWKES_PARAMS = {
    "mu_fast": 0.42099580293053274,
    "mu_slow": 0.48524131259737846,
    "H_base": 0.006935493256118603,
    "lambda_fast": 4.859223589602316,
    "lambda_slow": 1.9999982036603356,
}

CLASSIFIED_EVENTS_CSV = Path("classified_events_35.csv")


def pick_default_topics(seed: int = 42, k: int = 5) -> List[str]:
    """
    从 classified_events_35.csv 中抽取唯一 topic；若随机不可控，则取前 k 个。
    """
    if CLASSIFIED_EVENTS_CSV.exists():
        df = pd.read_csv(CLASSIFIED_EVENTS_CSV)
        if "topic" in df.columns:
            uniq = df["topic"].dropna().unique().tolist()
            if uniq:
                # 按要求优先取前 k 个（保持确定性），若不足则返回全部
                return uniq[: min(k, len(uniq))]
    return []


def build_agents(llm: LLMClient, topics: Optional[List[str]] = None) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}

    agents["BrandOfficial"] = Agent(
        name="BrandOfficial",
        role="brand_official",
        profile="品牌官方账号，目标是维护品牌形象并稳定舆论。",
        llm_client=llm,
        topics=topics,
    )

    for i in range(5):
        name = f"AngryUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="angry_user",
            profile="对负面事件非常愤怒，容易发表激烈批评。",
            llm_client=llm,
            topics=topics,
        )

    for i in range(5):
        name = f"NeutralUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="neutral_user",
            profile="对事件保持观望，容易被他人观点影响。",
            llm_client=llm,
            topics=topics,
        )

    for i in range(3):
        name = f"FanUser{i+1}"
        agents[name] = Agent(
            name=name,
            role="fan_user",
            profile="长期关注品牌，倾向于为品牌辩护。",
            llm_client=llm,
            topics=topics,
        )

    agents["Media1"] = Agent(
        name="Media1",
        role="media",
        profile="科技媒体账号，追求流量，也关心事实。",
        llm_client=llm,
        topics=topics,
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
    any_agent = next(iter(env.agents.values()))
    llm = any_agent.llm

    system = "你是一个科技媒体账号，语气略带煽动。"
    if topic:
        user = f"""写一条关于“{topic}”的爆料帖，可以质疑其中的风险或可信度，但不要太长，1~2 句话。"""
    else:
        user = """写一条关于“热点科技产品存在安全隐患”的爆料帖，可以质疑其可靠性，但不要太长，1~2 句话。"""
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
    T: int = 100,
    seed: int = 42,
    topics: Optional[List[str]] = None,
    request_delay: float = 0.0,
    hawkes_params: Optional[dict] = None,
    pr_strategy: Optional[str] = None,
):
    """
    运行多时间步模拟，返回环境、每步新增帖子列表、以及话题热度快照。
    """
    random.seed(seed)
    if not topics:
        topics = pick_default_topics(seed=seed, k=5)

    llm = LLMClient()
    agents = build_agents(llm, topics=topics)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G, topics=topics, hawkes_params=hawkes_params or BEST_HAWKES_PARAMS)

    # 选择公关策略
    strategy = None
    if pr_strategy == "S1":
        strategy = DelayedApologyStrategy(brand_name="BrandOfficial")
    elif pr_strategy == "S2":
        strategy = FastClarifyStrategy(brand_name="BrandOfficial")
    elif pr_strategy == "S0":
        strategy = DoNothingStrategy(brand_name="BrandOfficial")

    # 初始爆料：为每个话题种子一条（若未提供话题，则发一条默认）
    if topics:
        for tp in topics:
            inject_initial_rumor(env, topic=tp)
    else:
        inject_initial_rumor(env, topic=None)

    steps = []
    heat_history = []
    for _ in range(1, T + 1):
        new_posts = env.step(pr_strategy=strategy, request_delay=request_delay)
        steps.append(new_posts)
        if env.topic_manager:
            snapshot = {"time": env.t}
            for topic in env.topic_manager.topics:
                snapshot[topic] = env.topic_manager.get_heat(topic)
            heat_history.append(snapshot)
    return env, steps, heat_history


def create_simulation_instance(strategy_name: str, seed: int = 42):
    """创建一套完整的模拟实例：llm + agents + graph + env + pr_strategy"""
    random.seed(seed)

    default_topics = pick_default_topics(seed=seed, k=5)
    llm = LLMClient()
    agents = build_agents(llm, topics=default_topics)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G, topics=default_topics, hawkes_params=BEST_HAWKES_PARAMS)

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


def run_once(strategy_name: str = "S0", T: int = 100, seed: int = 42):
    random.seed(seed)

    default_topics = pick_default_topics(seed=seed, k=5)
    llm = LLMClient()
    agents = build_agents(llm, topics=default_topics)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G, topics=default_topics, hawkes_params=BEST_HAWKES_PARAMS)

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
