"""
Narrative Agent for algorithmic trading based on market narratives.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame

from src.narrative_agent.config import NarrativeAgentConfig
from src.narrative_agent.position_manager import PositionManager
from src.narrative_agent.transaction_costs import TransactionCostModel
from src.narrative_agent.data import DataFetcher
from src.narrative_agent.strategy import CompositeStrategy, get_market_condition


class NarrativeAgent:
    """
    A Narrative Agent that trades based on historical narrative patterns.
    """

    def __init__(
        self,
        config: NarrativeAgentConfig,
        api_key: str,
        use_cache: bool = True,
        cache_dir: str = ".narrative_cache",
    ):
        """
        Initialize the Narrative Agent.
        """
        self.config = config
        self.api_key = api_key

        # Create agent ID for persistence
        self.id = f"{config.ticker}_{config.look_back_period}_{config.hold_period}_{config.transaction_cost}"

        # Initialize transaction cost model if using enhanced costs
        transaction_cost_model = None
        if config.use_enhanced_costs:
            transaction_cost_model = TransactionCostModel(
                gas_fee_usd=config.gas_fee_usd,
                amm_liquidity_usd=config.amm_liquidity_usd,
                position_size_usd=config.position_size_usd,
            )
            print("Using enhanced transaction cost model:")
            print(f"  - Gas fee: ${config.gas_fee_usd} per transaction")
            print(f"  - AMM liquidity: ${config.amm_liquidity_usd:,.0f}")
            print(f"  - Position size: ${config.position_size_usd:,.0f}")

        # Initialize components
        self.position_manager = PositionManager(
            config.hold_period,
            config.transaction_cost,
            config.stop_loss,
            config.stop_gain,
            transaction_cost_model=transaction_cost_model,
            use_enhanced_costs=config.use_enhanced_costs,
        )
        self.data_fetcher = DataFetcher(
            api_key, use_cache=use_cache, cache_dir=cache_dir
        )

        # Initialize strategy
        self.strategy = CompositeStrategy()

        # Storage for narratives and prices
        self.narratives_stored: List[Dict[str, Any]] = []
        self.prices_stored: List[Tuple[str, float]] = []

        print(f"NarrativeAgent initialized for {config.ticker}")
        if use_cache:
            print(f"  - Data caching enabled (dir: {cache_dir})")
        else:
            print("  - Data caching disabled")

    def update_narratives_and_prices(self, start_date: str, end_date: str) -> None:
        """
        Update stored narratives and prices for the given date range.
        """
        # Fetch new narratives
        narratives = self.data_fetcher.get_narratives(
            self.config.ticker, start_date, end_date
        )
        self.narratives_stored.extend(narratives)

        # Sort and deduplicate narratives
        self.narratives_stored = sorted(
            self.narratives_stored, key=lambda x: x["timestamp"]
        )
        seen_ids = set()
        unique_narratives = []
        for narrative in self.narratives_stored:
            if narrative["ID"] not in seen_ids:
                unique_narratives.append(narrative)
                seen_ids.add(narrative["ID"])
        self.narratives_stored = unique_narratives

        print(
            f"Updated narratives for {self.config.ticker} from {start_date} to {end_date}: Complete"
        )

        # Fetch new prices
        prices = self.data_fetcher.get_prices(self.config.ticker, start_date, end_date)
        self.prices_stored.extend(prices)

        # Sort and deduplicate prices
        self.prices_stored = sorted(self.prices_stored, key=lambda x: x[0])
        seen_timestamps = set()
        unique_prices = []
        for timestamp, price in self.prices_stored:
            if (timestamp, price) not in seen_timestamps:
                unique_prices.append((timestamp, price))
                seen_timestamps.add((timestamp, price))
        self.prices_stored = unique_prices

        print(
            f"Updated prices for {self.config.ticker} from {start_date} to {end_date}: Complete"
        )

    def search_similar_narratives(self, narrative_id: str) -> List[Dict[str, Any]]:
        """
        Search for similar narratives prior to the given narrative ID.
        """
        similar_narratives: List[Dict[str, Any]] = []

        # Find the reference narrative
        narrative_keywords = []
        narrative_timestamp = ""
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_keywords = narrative.get("pattern_keywords", [])
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_keywords or not narrative_timestamp:
            return similar_narratives

        # Find similar narratives based on keyword overlap
        for narrative in self.narratives_stored:
            # Skip if it's the same narrative or occurs after
            if narrative["timestamp"] >= narrative_timestamp:
                continue

            narrative_keywords_i = narrative.get("pattern_keywords", [])
            if not narrative_keywords_i:
                continue

            # Count common keywords (case-insensitive)
            narrative_keywords_lower = [kw.lower() for kw in narrative_keywords_i]
            common_count = sum(
                1 for kw in narrative_keywords if kw.lower() in narrative_keywords_lower
            )

            if common_count >= self.config.count_common_threshold:
                similar_narratives.append(narrative)

        return similar_narratives

    def get_price_at_timestamp(self, timestamp: str) -> Optional[float]:
        """
        Get the price at or before the given timestamp.
        """
        price = None
        for ts, p in reversed(self.prices_stored):
            if ts <= timestamp:
                price = p
                break
        return price

    def calculate_price_position(self, timestamp: str) -> Optional[float]:
        """
        Calculate the price position over the look-back period.
        """
        price = self.get_price_at_timestamp(timestamp)
        if price is None:
            return None

        # Get prices over the look-back period
        start_timestamp = (
            datetime.fromisoformat(timestamp)
            - timedelta(hours=self.config.look_back_period)
        ).isoformat()

        price_list: List[float] = []
        for ts, p in reversed(self.prices_stored):
            if ts <= timestamp and ts >= start_timestamp:
                price_list.append(p)

        if len(price_list) <= 1:
            return None

        price_range = max(price_list) - min(price_list)
        if price_range == 0:
            return None

        price_position = (price - min(price_list)) / price_range
        return price_position

    def calculate_price_return(self, timestamp: str) -> Optional[float]:
        """
        Calculate the price return over the hold period.
        """
        entry_price = self.get_price_at_timestamp(timestamp)
        if entry_price is None or entry_price == 0:
            return None

        # Get price after hold period
        exit_timestamp = (
            datetime.fromisoformat(timestamp) + timedelta(hours=self.config.hold_period)
        ).isoformat()

        exit_price = None
        for ts, p in self.prices_stored:
            if ts >= exit_timestamp:
                exit_price = p
                break

        if exit_price is None:
            return None

        return exit_price / entry_price - 1

    def generate_position(self, narrative_id: str) -> Optional[int]:
        """
        Generate a position signal using composite scoring approach.
        """
        # Create context for strategy
        context = {
            "narratives_stored": self.narratives_stored,
            "prices_stored": self.prices_stored,
            "config": self.config,
            "calculate_price_position": self.calculate_price_position,
            "calculate_price_return": self.calculate_price_return,
            "search_similar_narratives": self.search_similar_narratives,
            "get_price_at_timestamp": self.get_price_at_timestamp,
            "get_market_condition": lambda ts: get_market_condition(
                ts, self.prices_stored, self.get_price_at_timestamp
            ),
        }

        # Get composite score from strategy
        composite_score = self.strategy.calculate_composite_score(narrative_id, context)

        if composite_score is None:
            return None

        # Generate position based on composite score
        if composite_score > 0.25:  # Lowered threshold for more positions
            return 1
        elif composite_score < -0.25:
            return -1
        else:
            return 0

    def check_stop_conditions(self, timestamp: str) -> None:
        """
        Check for stop loss/gain conditions on active positions.
        """
        if self.position_manager.active_position is None:
            return

        # Get all prices between position entry and current timestamp
        entry_timestamp = self.position_manager.active_position.entry_timestamp

        for ts, price in self.prices_stored:
            # Skip prices before position entry
            if ts < entry_timestamp:
                continue

            # Stop checking after current timestamp
            if ts > timestamp:
                break

            # Check stop conditions at each price point
            self.position_manager.check_and_close_stop_conditions(ts, price)

            # If position was closed, no need to continue
            if self.position_manager.active_position is None:
                break

    def update(self, timestamp: str) -> None:
        """
        Main update function to be called periodically (hardcoded 24 hours):
            1. Checks stop conditions and closes positions if triggered
            2. Checks and closes expired positions
            3. Updates data if needed
            4. Analyzes new narratives and opens positions
        """
        end_date = timestamp
        start_date = (
            datetime.fromisoformat(end_date) - timedelta(hours=24)
        ).isoformat()

        # First check for stop conditions
        self.check_stop_conditions(timestamp)

        # Check if any active position needs to be closed due to expiry
        current_price = self.get_price_at_timestamp(timestamp)
        if current_price:
            self.position_manager.check_and_close_expired_position(
                timestamp, current_price
            )

        # Check if need to update data
        data_exists = any(start_date <= ts <= end_date for ts, _ in self.prices_stored)

        if not data_exists:
            self.update_narratives_and_prices(start_date, end_date)
        else:
            print(
                f"Update narratives for {self.config.ticker} from {start_date} to {end_date}: Skip"
            )
            print(
                f"Update prices for {self.config.ticker} from {start_date} to {end_date}: Skip"
            )

        # Process narratives in the current time window
        for narrative in self.narratives_stored:
            # Skip if outside time window
            if narrative["timestamp"] < start_date or narrative["timestamp"] > end_date:
                continue

            # Skip if no pattern
            if not narrative.get("pattern"):
                continue

            # Generate position signal
            position = self.generate_position(narrative["ID"])
            if position is None or position == 0:
                continue

            # Get entry price
            entry_price = self.get_price_at_timestamp(narrative["timestamp"])
            if entry_price is None:
                continue

            # Open position with configured position size
            self.position_manager.open_position(
                narrative,
                position,
                narrative["timestamp"],
                entry_price,
                position_value_usd=self.config.position_size_usd,
            )

    def finalize_positions(self, timestamp: str) -> None:
        """
        Finalize any open positions at a given timestamp.
        """
        final_price = self.get_price_at_timestamp(timestamp)
        if final_price:
            self.position_manager.finalize_positions(timestamp, final_price)

    def get_performance_dataframe(self) -> DataFrame:
        """
        Get a DataFrame with position history and performance metrics.
        """
        position_history = self.position_manager.get_position_history()
        if not position_history:
            return DataFrame()

        df = DataFrame(position_history)

        # Sort by entry timestamp
        df = df.sort_values("entry_timestamp")

        # Calculate cumulative returns
        df["cum_return"] = df["position_return"].cumsum()

        # Calculate Maximum Drawdown
        df["running_peak"] = df["cum_return"].cummax()
        df["drawdown"] = df["cum_return"] - df["running_peak"]
        df["max_drawdown"] = df["drawdown"].cummin()
        df = df.drop(columns=["running_peak", "drawdown"])

        # Calculate annualized volatility
        if len(df) > 1:
            period_std = df["position_return"].std(ddof=1)
            annualized_vol = period_std * np.sqrt(365 * 24 * 60)
            df["vol_annualized"] = annualized_vol
        else:
            df["vol_annualized"] = 0.0

        return df

    def get_transaction_cost_summary(self) -> Dict[str, Any]:
        """
        Get summary of transaction costs if using enhanced model.
        """
        if not self.config.use_enhanced_costs:
            return {"message": "Enhanced cost model not enabled"}

        cost_model = self.position_manager.transaction_cost_model
        if cost_model is None:
            return {"message": "Transaction cost model not initialized"}

        return cost_model.get_cost_metrics()

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self.data_fetcher.clear_cache()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        """
        return self.data_fetcher.get_cache_info()
