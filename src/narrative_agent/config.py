"""
Configuration for NarrativeAgent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NarrativeAgentConfig:
    """
    Configuration for a NarrativeAgent.
    """

    # Asset to trade
    ticker: str = "BTC"

    # Price position calculation
    look_back_period: int = 6
    """In hours."""

    # Position management
    hold_period: int = 1
    """In hours."""

    transaction_cost: int = 10
    """In basis points."""

    # Enhanced transaction cost parameters
    gas_fee_usd: float = 1.0
    """Gas cost per transaction in USD."""

    amm_liquidity_usd: float = 100_000_000.0
    """AMM pool TVL in USD."""

    position_size_usd: float = 10_000.0
    """Default position size in USD."""

    use_enhanced_costs: bool = True
    """Use enhanced cost model."""

    # Pattern matching
    count_common_threshold: int = 5
    """Minimum common keywords."""

    # Risk management
    stop_loss: Optional[float] = None
    """Stop loss percentage (e.g., 5 = 5%)."""

    stop_gain: Optional[float] = None
    """Stop gain percentage (e.g., 10 = 10%)."""

    def __post_init__(self):
        """
        Validate configuration.
        """
        if self.look_back_period <= 0:
            raise ValueError("look_back_period must be positive")
        if self.hold_period <= 0:
            raise ValueError("hold_period must be positive")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        if self.gas_fee_usd < 0:
            raise ValueError("gas_fee_usd must be non-negative")
        if self.amm_liquidity_usd <= 0:
            raise ValueError("amm_liquidity_usd must be positive")
        if self.position_size_usd <= 0:
            raise ValueError("position_size_usd must be positive")
        if self.count_common_threshold <= 0:
            raise ValueError("count_common_threshold must be positive")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive if set")
        if self.stop_gain is not None and self.stop_gain <= 0:
            raise ValueError("stop_gain must be positive if set")
