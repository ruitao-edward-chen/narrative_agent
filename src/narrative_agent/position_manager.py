"""
Position management for the NarrativeAgent.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


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
    ):
        """
        Initialize the position manager.
        """
        self.hold_period = hold_period
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.stop_gain = stop_gain
        self.active_position: Optional[Position] = None
        self.position_history: List[Position] = []

    def open_position(
        self,
        narrative: Dict[str, Any],
        position: int,
        timestamp: str,
        entry_price: float,
    ) -> None:
        """
        Open a new position after closing any existing position.
        """
        # Close existing position (if applicable).
        if self.active_position is not None:
            self.close_position(timestamp, entry_price, "override")

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
        )

        # Log the opening.
        print(
            f"Opening position: {position} at {timestamp[:19]} "
            f"(narrative: {narrative['ID'][:8]}...), "
            f"expected close: {self.active_position.expected_close_timestamp[:19]}"
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
        print(
            f"Closing position: {self.active_position.position} "
            f"from {self.active_position.entry_timestamp[:19]} "
            f"to {close_timestamp[:19]} "
            f"(reason: {close_reason}), "
            f"return: {position_return:.4%}"
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
            }

        winning_positions = sum(1 for r in returns if r > 0)

        return {
            "total_positions": len(self.position_history),
            "total_return": sum(returns),
            "avg_return": sum(returns) / len(returns),
            "win_rate": winning_positions / len(returns) if returns else 0.0,
            "max_return": max(returns),
            "min_return": min(returns),
        }

    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get position history as a list of dictionaries.
        """
        return [
            {
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
            for pos in self.position_history
        ]
