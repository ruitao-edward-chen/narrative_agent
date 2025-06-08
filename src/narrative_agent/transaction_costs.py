"""
Enhanced Transaction Cost Model for DeFi Trading.
"""

from typing import Dict, Any
from dataclasses import dataclass

from .amm_pool import AMMPool


@dataclass
class TransactionCostBreakdown:
    """
    Breakdown of transaction costs.
    """

    gas_fee_usd: float

    slippage_bps: float

    slippage_usd: float

    protocol_fee_usd: float

    total_cost_usd: float

    total_cost_bps: float

    effective_price: float

    price_impact_pct: float


class TransactionCostModel:
    """
    Comprehensive transaction cost model for DeFi trading.
    """

    def __init__(
        self,
        gas_fee_usd: float = 50.0,
        amm_liquidity_usd: float = 100_000_000.0,
        amm_fee_tier: float = 0.003,
        position_size_usd: float = 10_000.0,
    ):
        """
        Initialize transaction cost model.
        """
        self.gas_fee_usd = gas_fee_usd
        self.position_size_usd = position_size_usd
        self.amm_pool = AMMPool(amm_liquidity_usd, amm_fee_tier)

        self.total_gas_paid = 0.0
        self.total_slippage_paid = 0.0
        self.total_fees_paid = 0.0
        self.transaction_count = 0

    def calculate_transaction_costs(
        self,
        position_value_usd: float,
        current_price: float,
        is_buy: bool,
        is_entry: bool = True,
    ) -> TransactionCostBreakdown:
        """
        Calculate comprehensive transaction costs for a trade.
        Note: Protocol fees are included in the slippage calculation by the AMM.
        """
        # Ensure pool is synced to current price
        self.amm_pool.sync_to_price(current_price)

        # Calculate total slippage (includes protocol fee)
        total_slippage_bps = self.amm_pool.calculate_slippage_bps(
            position_value_usd, current_price, is_buy
        )

        # This is an approximation for reporting purposes
        protocol_fee_bps = self.amm_pool.fee_tier * 10000

        # Pure price impact (excluding protocol fee) as an approximation
        price_impact_bps = max(0, total_slippage_bps - protocol_fee_bps)

        # Calculate USD amounts
        total_slippage_usd = position_value_usd * (total_slippage_bps / 10000)
        protocol_fee_usd = position_value_usd * (protocol_fee_bps / 10000)

        # Gas fee is fixed per transaction
        gas_fee = self.gas_fee_usd

        # Total costs
        total_cost_usd = gas_fee + total_slippage_usd
        total_cost_bps = (total_cost_usd / position_value_usd) * 10000

        # Calculate effective price after slippage
        if is_buy:
            effective_price = current_price * (1 + total_slippage_bps / 10000)
        else:
            effective_price = current_price * (1 - total_slippage_bps / 10000)

        # Calculate price impact percentage
        price_impact_pct = (price_impact_bps / 10000) * 100

        # Update tracking metrics
        self.transaction_count += 1
        self.total_gas_paid += gas_fee
        self.total_slippage_paid += total_slippage_usd
        self.total_fees_paid += protocol_fee_usd

        return TransactionCostBreakdown(
            gas_fee_usd=gas_fee,
            slippage_bps=total_slippage_bps,
            slippage_usd=total_slippage_usd,
            protocol_fee_usd=protocol_fee_usd,
            total_cost_usd=total_cost_usd,
            total_cost_bps=total_cost_bps,
            effective_price=effective_price,
            price_impact_pct=price_impact_pct,
        )

    def calculate_round_trip_costs(
        self,
        position_value_usd: float,
        entry_price: float,
        exit_price: float,
        position_type: int,
    ) -> Dict[str, Any]:
        """
        Calculate total costs for a round-trip trade (entry + exit).
        """
        # Entry costs
        is_buy_entry = position_type == 1
        entry_costs = self.calculate_transaction_costs(
            position_value_usd, entry_price, is_buy_entry, is_entry=True
        )

        # Exit costs (opposite of entry)
        is_buy_exit = position_type == -1
        exit_costs = self.calculate_transaction_costs(
            position_value_usd, exit_price, is_buy_exit, is_entry=False
        )

        # Calculate position P&L before costs
        if position_type == 1:  # Long
            price_return = (exit_price - entry_price) / entry_price
        else:  # Short
            price_return = (entry_price - exit_price) / entry_price

        gross_pnl = position_value_usd * price_return

        # Net PnL after costs
        total_costs = entry_costs.total_cost_usd + exit_costs.total_cost_usd
        net_pnl = gross_pnl - total_costs

        return {
            "entry_costs": entry_costs,
            "exit_costs": exit_costs,
            "total_cost_usd": total_costs,
            "total_cost_bps": (total_costs / position_value_usd) * 10000,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "cost_as_pct_of_gross": (
                (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
            ),
        }

    def get_cost_metrics(self) -> Dict[str, float]:
        """
        Get summary metrics for all transaction costs.
        """
        # Don't add fees_paid to total since it's already included in slippage
        total_costs = self.total_gas_paid + self.total_slippage_paid
        position_count = self.transaction_count // 2

        return {
            "transaction_count": self.transaction_count,
            "position_count": position_count,
            "total_gas_paid": self.total_gas_paid,
            "total_slippage_paid": self.total_slippage_paid,  # This includes protocol fees
            "total_fees_paid": self.total_fees_paid,  # Keep for display breakdown
            "total_costs": total_costs,  # Fixed: no longer double-counting protocol fees
            "avg_gas_per_tx": self.total_gas_paid / max(1, self.transaction_count),
            "avg_slippage_per_tx": self.total_slippage_paid
            / max(1, self.transaction_count),
            "avg_total_cost_per_tx": total_costs / max(1, self.transaction_count),
            "avg_cost_per_position": total_costs / max(1, position_count),
            "gas_pct_of_total": (
                (self.total_gas_paid / total_costs * 100) if total_costs > 0 else 0
            ),
            "slippage_pct_of_total": (
                (self.total_slippage_paid / total_costs * 100) if total_costs > 0 else 0
            ),
            "fees_pct_of_total": (
                (self.total_fees_paid / total_costs * 100) if total_costs > 0 else 0
            ),
        }

    def estimate_break_even_move(
        self, position_value_usd: float, current_price: float, position_type: int
    ) -> float:
        """
        Calculate the minimum price move needed to break even after costs.
        """
        # Calculate round trip costs at current price
        costs = self.calculate_round_trip_costs(
            position_value_usd, current_price, current_price, position_type
        )

        # Required return to cover costs
        required_return = costs["total_cost_usd"] / position_value_usd

        return required_return * 100
