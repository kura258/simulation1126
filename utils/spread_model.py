import numpy as np


class HawkesProcess:
    """
    霍克斯过程：自激事件过程，用于模拟“越多事件越容易继续发生”。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, mu: float = 0.1):
        self.alpha = alpha  # 自激强度
        self.beta = beta    # 衰减因子
        self.mu = mu        # 基线强度
        self.events = []    # 已记录的事件发生时刻

    def simulate_event(self, current_time: float) -> bool:
        """
        在 current_time 时刻判断是否发生事件；若发生则记录并返回 True。
        """
        intensity = self.mu + sum(
            self.alpha * np.exp(-self.beta * (current_time - t)) for t in self.events
        )
        if np.random.random() < intensity:
            self.events.append(current_time)
            return True
        return False

    def get_event_count(self) -> int:
        """返回当前累计事件数量。"""
        return len(self.events)
