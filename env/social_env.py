from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import time

import networkx as nx


@dataclass
class Post:
    id: int
    author: str
    text: str
    sentiment: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    tag: str        # "rumor", "official", "user", ...
    time_step: int
    target_post_id: Optional[int] = None
    topic: Optional[str] = None


class TopicManager:
    """
    维护话题热度，混合基础统计与简化的 Hawkes 记忆项。
    """

    def __init__(self, topics: Sequence[str], hawkes_params: Optional[Dict[str, float]] = None):
        params = hawkes_params or {}
        self.alpha_v = params.get("alpha_v", params.get("alpha", 1.0))
        self.beta_c = params.get("beta_c", params.get("beta", 0.5))
        self.gamma_r = params.get("gamma_r", params.get("gamma", 0.3))
        # 兼容训练得到的双衰减参数，退化为单核使用 mu_fast / lambda_fast 近似
        self.mu = params.get("mu", params.get("mu_fast", 0.1))
        self.decay = params.get("decay", params.get("lambda", params.get("lambda_fast", 0.5)))

        self.topics: Dict[str, Dict[str, Any]] = {
            topic: {
                "heat": 0.0,
                "posts": [],
                "events": [],
                "per_step": {},  # t -> {"V": count, "C": 评论数, "R": reach}
            } for topic in topics
        }

    def _compute_heat(self, topic: str, current_time: int) -> float:
        tdata = self.topics[topic]
        stats = tdata["per_step"].get(current_time, {"V": 0, "C": 0, "R": 0})
        V = stats["V"]
        C = stats["C"]
        R = stats["R"]

        base = (
            self.alpha_v * math.log(V + 1)
            + self.beta_c * math.log(C + 1)
            + self.gamma_r * math.log(R + 1)
        )
        hawkes = 0.0
        for ti in tdata["events"]:
            if ti < current_time:
                hawkes += self.mu * math.exp(-self.decay * (current_time - ti))
        return base + hawkes

    def add_post(self, topic: str, post_content: str, current_time: int = 0, reach: float = 0.0) -> None:
        """
        记录新增帖子并刷新热度；reach 表示影响范围，可简单累加。
        """
        if topic not in self.topics:
            return
        tdata = self.topics[topic]
        tdata["posts"].append(post_content)
        tdata["events"].append(current_time)
        step_stats = tdata["per_step"].setdefault(current_time, {"V": 0, "C": 0, "R": 0})
        step_stats["V"] += 1
        step_stats["R"] += reach

        tdata["heat"] = self._compute_heat(topic, current_time)

    def get_heat(self, topic: str) -> float:
        """获取当前热度；未知话题返回 0。"""
        return self.topics.get(topic, {}).get("heat", 0.0)


class SocialEnv:
    """
    社交环境：
    - G：关注关系图
    - agents：name -> Agent
    - posts：历史帖子列表
    - topic_manager：记录各话题热度
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        graph: nx.DiGraph,
        topics: Optional[Sequence[str]] = None,
        hawkes_params: Optional[Dict[str, float]] = None,
    ):
        self.agents = agents
        self.G = graph
        self.posts: List[Post] = []
        self.t = 0
        self._next_post_id = 1
        self._topics = list(topics) if topics else []
        self._hawkes_params = hawkes_params
        self.topic_manager: Optional[TopicManager] = (
            TopicManager(self._topics, hawkes_params) if self._topics else None
        )

    def reset(self):
        self.posts = []
        self.t = 0
        self._next_post_id = 1
        if self._topics:
            self.topic_manager = TopicManager(self._topics, self._hawkes_params)

    def _compute_reach(self, author: str) -> int:
        """简单地以关注入度作为传播影响力近似。"""
        return self.G.in_degree(author)

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
        topic: Optional[str] = None,
    ):
        post = Post(
            id=self._next_post_id,
            author=author,
            text=text,
            sentiment=sentiment,
            tag=tag,
            time_step=self.t,
            target_post_id=target_post_id,
            topic=topic,
        )
        self._next_post_id += 1
        self.posts.append(post)

        if self.topic_manager and topic:
            reach = self._compute_reach(author)
            self.topic_manager.add_post(topic, text, current_time=self.t, reach=reach)

    def step(self, pr_strategy=None, request_delay: float = 0.0):
        """推进一个时间步，驱动所有 Agent 发言。"""
        self.t += 1
        new_posts: List[Post] = []
        observed = [
            {
                "id": p.id,
                "author": p.author,
                "text": p.text,
                "summary": p.text,
                "sentiment": p.sentiment,
                "tag": p.tag,
                "topic": p.topic,
            }
            for p in self.posts
            if p.time_step == self.t - 1
        ]

        for name, agent in self.agents.items():
            if request_delay > 0:
                time.sleep(request_delay)

            if pr_strategy and name == "BrandOfficial":
                action = pr_strategy.decide_brand_action(self.t, agent, observed)
            else:
                action = agent.decide_social_action(self.t, observed, environment=self)

            if action is None:
                continue

            act_type = action.get("action") or action.get("type")
            if act_type == "post":
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag=action.get("tag", "user"),
                    topic=action.get("topic"),
                )
                new_posts.append(self.posts[-1])
            elif act_type == "retweet":
                target_id = action.get("target_post_id")
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag="retweet",
                    target_post_id=target_id,
                    topic=action.get("topic"),
                )
                new_posts.append(self.posts[-1])
            else:
                continue
        return new_posts
