import numpy as np


class HawkesProcess:
    """
    Simple Hawkes process to model self-exciting events.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, mu: float = 0.1):
        self.alpha = alpha  # excitation strength
        self.beta = beta    # decay factor
        self.mu = mu        # baseline intensity
        self.events = []    # recorded event times

    def simulate_event(self, current_time: float) -> bool:
        """
        Simulate whether an event happens at current_time.
        Returns True if an event is added.
        """
        intensity = self.mu + sum(
            self.alpha * np.exp(-self.beta * (current_time - t)) for t in self.events
        )
        if np.random.random() < intensity:
            self.events.append(current_time)
            return True
        return False

    def get_event_count(self) -> int:
        """Return the number of events that have occurred."""
        return len(self.events)
