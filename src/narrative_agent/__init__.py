"""
Narrative Agent for algorithmic trading based on market narratives.

This module provides tools for:
- Pattern-based narrative analysis
- DeFi transaction cost modeling
- Automated position management
- Risk-aware trading execution
"""

from .agent import NarrativeAgent
from .config import NarrativeAgentConfig
from .position_manager import PositionManager
from .transaction_costs import TransactionCostModel, TransactionCostBreakdown
from .amm_pool import AMMPool

__all__ = [
    "NarrativeAgent",
    "NarrativeAgentConfig",
    "PositionManager",
    "TransactionCostModel",
    "TransactionCostBreakdown",
    "AMMPool",
]

__version__ = "1.1.0"  # Updated for enhanced transaction cost model
