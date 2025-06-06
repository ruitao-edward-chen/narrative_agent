"""
Config for NarrativeAgent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NarrativeAgentConfig:
    """
    Config for NarrativeAgent.
    """

    ticker: str

    look_back_period: int
    """ In hours. """

    hold_period: int
    """ In hours. """

    transaction_cost: int
    """ In basis points. """

    count_common_threshold: int

    stop_loss: Optional[float] = None

    stop_gain: Optional[float] = None

    def __post_init__(self):
        """
        Validate configuration parameters.
        """
        if self.look_back_period <= 0:
            raise ValueError("look_back_period must be positive")
        if self.hold_period <= 0:
            raise ValueError("hold_period must be positive")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative")
        if self.count_common_threshold <= 0:
            raise ValueError("count_common_threshold must be positive")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive")
        if self.stop_gain is not None and self.stop_gain <= 0:
            raise ValueError("stop_gain must be positive")
