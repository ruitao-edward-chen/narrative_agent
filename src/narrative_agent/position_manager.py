"""
Position management for the NarrativeAgent.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.narrative_agent.transaction_costs import (
    TransactionCostModel,
    TransactionCostBreakdown,
)


@dataclass
class Position:
    """
    A trading position.
    """

    entry_timestamp: str

    entry_price: float

    position: int

    narrative_id: str

    narrative_summary: str

    expected_close_timestamp: str

    close_timestamp: Optional[str] = None

    close_price: Optional[float] = None

    close_reason: Optional[str] = None

    position_return: Optional[float] = None

    entry_costs: Optional[TransactionCostBreakdown] = None

    exit_costs: Optional[TransactionCostBreakdown] = None

    position_value_usd: Optional[float] = None


class PositionManager:
    """
    Manages trading positions with explicit entry/exit logic.
    """

    def __init__(
        self,
        hold_period: int,
        transaction_cost: int,
        stop_loss: Optional[float] = None,
        stop_gain: Optional[float] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        use_enhanced_costs: bool = True,
    ):
        """
        Initialize the position manager.
        """
        self.hold_period = hold_period
        self.stop_loss = stop_loss
        self.stop_gain = stop_gain
        self.active_position: Optional[Position] = None
        self.position_history: List[Position] = []
        # For non-enhanced mode.
        self.transaction_cost = transaction_cost
        # For enhanced mode.
        self.transaction_cost_model = transaction_cost_model
        self.use_enhanced_costs = (
            use_enhanced_costs and transaction_cost_model is not None
        )

    def open_position(
        self,
        narrative: Dict[str, Any],
        position: int,
        timestamp: str,
        entry_price: float,
        position_value_usd: Optional[float] = None,
    ) -> None:
        """
        Open a new position after closing any existing position.
        """
        # Close existing position (if applicable).
        if self.active_position is not None:
            self.close_position(timestamp, entry_price, "override")

        # Determine position value
        if position_value_usd is None and self.transaction_cost_model:
            position_value_usd = self.transaction_cost_model.position_size_usd

        # Calculate entry costs if using enhanced model
        entry_costs = None
        if self.use_enhanced_costs and position_value_usd:
            is_buy = position == 1  # Long = buy, Short = sell
            entry_costs = self.transaction_cost_model.calculate_transaction_costs(
                position_value_usd, entry_price, is_buy, is_entry=True
            )

        # Create new position.
        self.active_position = Position(
            entry_timestamp=timestamp,
            entry_price=entry_price,
            position=position,
            narrative_id=narrative["ID"],
            narrative_summary=narrative["summary"],
            expected_close_timestamp=(
                datetime.fromisoformat(timestamp) + timedelta(hours=self.hold_period)
            ).isoformat(),
            entry_costs=entry_costs,
            position_value_usd=position_value_usd,
        )

        # Log the opening.
        cost_info = ""
        if entry_costs:
            cost_info = f", costs: {entry_costs.total_cost_bps:.1f}bps"

        print(
            f"Opening position: {position} at {timestamp[:19]} "
            f"(narrative: {narrative['ID'][:8]}...), "
            f"expected close: {self.active_position.expected_close_timestamp[:19]}"
            f"{cost_info}"
        )

    def close_position(
        self, close_timestamp: str, close_price: float, close_reason: str
    ) -> None:
        """
        Close the active position and calculate returns.
        """
        if self.active_position is None:
            return

        # Calculate return
        if self.active_position.entry_price == 0:
            return

        price_return = close_price / self.active_position.entry_price - 1

        # Calculate costs and net return
        if self.use_enhanced_costs and self.active_position.position_value_usd:
            # Calculate exit costs
            is_buy = (
                self.active_position.position == -1
            )  # Exit long = sell, Exit short = buy
            exit_costs = self.transaction_cost_model.calculate_transaction_costs(
                self.active_position.position_value_usd,
                close_price,
                is_buy,
                is_entry=False,
            )

            # Calculate net position return after all costs
            gross_return = self.active_position.position * price_return
            total_cost_bps = (
                self.active_position.entry_costs.total_cost_bps
                + exit_costs.total_cost_bps
            )
            position_return = gross_return - total_cost_bps / 10000

            self.active_position.exit_costs = exit_costs
        else:
            # Legacy calculation
            position_return = (
                self.active_position.position * price_return
                - 2 * self.transaction_cost * 0.0001
            )

        # Update position with closing info
        self.active_position.close_timestamp = close_timestamp
        self.active_position.close_price = close_price
        self.active_position.close_reason = close_reason
        self.active_position.position_return = position_return

        # Log the closing
        cost_info = ""
        if self.use_enhanced_costs and self.active_position.exit_costs:
            total_costs = (
                self.active_position.entry_costs.total_cost_usd
                + self.active_position.exit_costs.total_cost_usd
            )
            cost_info = f", total costs: ${total_costs:.2f}"

        print(
            f"Closing position: {self.active_position.position} "
            f"from {self.active_position.entry_timestamp[:19]} "
            f"to {close_timestamp[:19]} "
            f"(reason: {close_reason}), "
            f"return: {position_return:.4%}"
            f"{cost_info}"
        )

        # Move to history
        self.position_history.append(self.active_position)
        self.active_position = None

    def check_and_close_stop_conditions(
        self, current_timestamp: str, current_price: float
    ) -> None:
        """
        Check if the active position has hit stop loss or stop gain thresholds.
        """
        if self.active_position is None:
            return

        # Calculate current return
        if self.active_position.entry_price == 0:
            return

        price_return = current_price / self.active_position.entry_price - 1
        position_return = self.active_position.position * price_return

        # Check stop loss
        if self.stop_loss is not None and position_return <= -self.stop_loss / 100:
            self.close_position(current_timestamp, current_price, "stop_loss")
            return

        # Check stop gain
        if self.stop_gain is not None and position_return >= self.stop_gain / 100:
            self.close_position(current_timestamp, current_price, "stop_gain")
            return

    def check_and_close_expired_position(
        self, current_timestamp: str, current_price: float
    ) -> None:
        """
        Check if the active position has exceeded its hold period and close it.
        """
        if self.active_position is None:
            return

        if current_timestamp >= self.active_position.expected_close_timestamp:
            self.close_position(
                self.active_position.expected_close_timestamp,
                current_price,
                "hold_period_expired",
            )

    def finalize_positions(self, final_timestamp: str, final_price: float) -> None:
        """
        Close any remaining active position at the final timestamp.
        Useful for end of backtest or when stopping the agent.
        """
        if self.active_position is not None:
            self.close_position(final_timestamp, final_price, "finalized")

    def get_position_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics for all positions.
        """
        if not self.position_history:
            return {
                "total_positions": 0,
                "total_return": 0.0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "max_return": 0.0,
                "min_return": 0.0,
                "total_costs_usd": 0.0,
            }

        returns = [
            p.position_return
            for p in self.position_history
            if p.position_return is not None
        ]

        if not returns:
            return {
                "total_positions": len(self.position_history),
                "total_return": 0.0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "max_return": 0.0,
                "min_return": 0.0,
                "total_costs_usd": 0.0,
            }

        winning_positions = sum(1 for r in returns if r > 0)

        # Calculate total costs if using enhanced model
        total_costs_usd = 0.0
        if self.use_enhanced_costs:
            for pos in self.position_history:
                if pos.entry_costs and pos.exit_costs:
                    total_costs_usd += (
                        pos.entry_costs.total_cost_usd + pos.exit_costs.total_cost_usd
                    )

        return {
            "total_positions": len(self.position_history),
            "total_return": sum(returns),
            "avg_return": sum(returns) / len(returns),
            "win_rate": winning_positions / len(returns) if returns else 0.0,
            "max_return": max(returns),
            "min_return": min(returns),
            "total_costs_usd": total_costs_usd,
        }

    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get position history as a list of dictionaries.
        """
        history = []
        for pos in self.position_history:
            position_dict = {
                "entry_timestamp": pos.entry_timestamp,
                "entry_price": pos.entry_price,
                "position": pos.position,
                "narrative_id": pos.narrative_id,
                "narrative_summary": pos.narrative_summary,
                "expected_close_timestamp": pos.expected_close_timestamp,
                "close_timestamp": pos.close_timestamp,
                "close_price": pos.close_price,
                "close_reason": pos.close_reason,
                "position_return": pos.position_return,
            }

            # Add enhanced cost information if available
            if self.use_enhanced_costs and pos.entry_costs and pos.exit_costs:
                position_dict.update(
                    {
                        "position_value_usd": pos.position_value_usd,
                        "entry_gas_fee": pos.entry_costs.gas_fee_usd,
                        "entry_slippage_bps": pos.entry_costs.slippage_bps,
                        "entry_total_cost_bps": pos.entry_costs.total_cost_bps,
                        "exit_gas_fee": pos.exit_costs.gas_fee_usd,
                        "exit_slippage_bps": pos.exit_costs.slippage_bps,
                        "exit_total_cost_bps": pos.exit_costs.total_cost_bps,
                        "total_cost_usd": (
                            pos.entry_costs.total_cost_usd
                            + pos.exit_costs.total_cost_usd
                        ),
                    }
                )

            history.append(position_dict)

        return history
