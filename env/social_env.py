from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx

from utils.spread_model import HawkesProcess


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
    Track topics, their heat, and associated posts using a Hawkes process per topic.
    """

    def __init__(self, topics: Sequence[str], hawkes_params: Optional[Dict[str, float]] = None):
        params = hawkes_params or {}
        self.topics: Dict[str, Dict[str, Any]] = {
            topic: {"heat": 0, "posts": []} for topic in topics
        }
        self.processes: Dict[str, HawkesProcess] = {
            topic: HawkesProcess(**params) for topic in topics
        }

    def add_post(self, topic: str, post_content: str, current_time: float = 0.0) -> None:
        """
        Add a post under a topic, bump heat, and update Hawkes-triggered spread.
        """
        if topic not in self.topics:
            return
        self.topics[topic]["posts"].append(post_content)
        self.topics[topic]["heat"] += 1

        process = self.processes.get(topic)
        if process and process.simulate_event(current_time):
            # Hawkes-triggered additional heat
            self.topics[topic]["heat"] += 1

    def get_heat(self, topic: str) -> int:
        """
        Return current heat for a topic.
        """
        return self.topics.get(topic, {}).get("heat", 0)


class SocialEnv:
    """
    Social media simulation environment.
    - G: directed follow graph
    - agents: name -> Agent
    - posts: history of posts
    - t: current time step
    - topic_manager: manage topic heat and posts
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

    # ---- internal helpers ----

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> Post:
        p = Post(
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
        self.posts.append(p)

        if self.topic_manager and topic:
            self.topic_manager.add_post(topic, text, current_time=self.t)

        return p

    def record_topic_interaction(self, topic: str, content: str = "") -> None:
        """
        Allow external callers (e.g., agents) to bump topic heat.
        """
        if self.topic_manager:
            self.topic_manager.add_post(topic, content or f"interaction on {topic}", current_time=self.t)

    def get_topic_heat(self, topic: str) -> int:
        """
        Fetch heat for a given topic. Returns 0 if topic is unknown.
        """
        if not self.topic_manager:
            return 0
        return self.topic_manager.get_heat(topic)

    def get_visible_posts_for(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        At time step t, agent_name can see posts from followed agents at t-1.
        """
        following = list(self.G.successors(agent_name))
        recent_posts = [p for p in self.posts if p.author in following and p.time_step == self.t - 1]
        result = []
        for p in recent_posts:
            result.append({
                "id": p.id,
                "author": p.author,
                "text": p.text,
                "summary": p.text[:50],
                "sentiment": p.sentiment,
                "tag": p.tag,
                "topic": p.topic,
            })
        return result

    # ---- core: advance one step ----

    def step(self, pr_strategy=None):
        """
        Execute one time step:
        1) Brand agent acts first via PR strategy.
        2) Other agents act based on observed posts.
        """
        self.t += 1
        new_posts: List[Post] = []

        # 1) brand official
        if pr_strategy is not None and hasattr(pr_strategy, "brand_name"):
            brand_name = pr_strategy.brand_name
            brand_agent = self.agents[brand_name]
            observed = self.get_visible_posts_for(brand_name)
            brand_action = pr_strategy.decide_brand_action(
                t=self.t,
                agent=brand_agent,
                observed_posts=observed,
            )
            if brand_action and brand_action["action"] != "silent":
                p = self._add_post(
                    author=brand_name,
                    text=brand_action["post_text"],
                    sentiment=brand_action.get("sentiment", "NEUTRAL"),
                    tag="official",
                    target_post_id=brand_action.get("target_post_id"),
                    topic=brand_action.get("topic"),
                )
                new_posts.append(p)
                brand_agent.observe(f"I posted an official statement at t={self.t}: {p.text}")

        # 2) other agents
        for name, agent in self.agents.items():
            if pr_strategy is not None and name == getattr(pr_strategy, "brand_name", None):
                continue

            observed = self.get_visible_posts_for(name)
            if not observed:
                continue

            action = agent.decide_social_action(self.t, observed)
            if action["action"] == "silent":
                continue

            p = self._add_post(
                author=name,
                text=action["post_text"],
                sentiment=action["sentiment"],
                tag="user",
                target_post_id=action.get("target_post_id"),
                topic=action.get("topic"),
            )
            new_posts.append(p)
            agent.observe(f"I posted on social media at t={self.t}: {p.text}")

        return new_posts
