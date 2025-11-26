from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from .agent import Agent


class MultiAgentSystem:
    """
    Simple container to manage multiple agents via a provided list or factory.
    """

    def __init__(
        self,
        agent_count: int = 0,
        topics: Optional[Sequence[str]] = None,
        agents: Optional[Iterable[Agent]] = None,
        agent_factory: Optional[Callable[[int, List[str]], Agent]] = None,
    ) -> None:
        self.topics = list(topics) if topics else []
        self.agents: List[Agent] = []

        if agents is not None:
            self.agents.extend(agents)
        elif agent_count > 0:
            if agent_factory is None:
                raise ValueError("Provide agent_factory or agents to build the system")
            for idx in range(agent_count):
                self.agents.append(agent_factory(idx, self.topics))

        if not self.agents:
            raise ValueError("MultiAgentSystem requires at least one Agent instance")

    def run_simulation_step(self, environment) -> None:
        """
        Drive each agent to interact with the environment once.
        """
        for agent in self.agents:
            agent.interact(environment)
