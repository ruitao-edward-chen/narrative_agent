"""
Narrative Agent for algorithmic trading based on market narratives.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame

from .config import NarrativeAgentConfig
from .position_manager import PositionManager
from .transaction_costs import TransactionCostModel
from .data import DataFetcher


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
        # Get composite score from multiple signals
        composite_score = self._calculate_composite_score(narrative_id)

        if composite_score is None:
            return None

        # Generate position based on composite score
        if composite_score > 0.25:  # Lowered threshold for more positions
            return 1
        elif composite_score < -0.25:
            return -1
        else:
            return 0

    def _calculate_composite_score(self, narrative_id: str) -> Optional[float]:
        """
        Enhanced composite score with dynamic weight adjustment.
        """
        # Get all signals including new ones
        signals = {}

        # Original signals
        pattern_signal = self._get_pattern_signal(narrative_id)
        if pattern_signal is not None:
            signals["pattern"] = pattern_signal

        sentiment_event_signal = self._get_sentiment_event_signal(narrative_id)
        if sentiment_event_signal is not None:
            signals["sentiment_event"] = sentiment_event_signal

        price_momentum_signal = self._get_price_momentum_signal(narrative_id)
        if price_momentum_signal is not None:
            signals["price_momentum"] = price_momentum_signal

        sentiment_momentum_signal = self._get_sentiment_momentum_signal(narrative_id)
        if sentiment_momentum_signal is not None:
            signals["sentiment_momentum"] = sentiment_momentum_signal

        # New enhanced signals
        volatility_signal = self._get_volatility_regime_signal(narrative_id)
        if volatility_signal is not None:
            signals["volatility_regime"] = volatility_signal

        clustering_signal = self._get_narrative_clustering_signal(narrative_id)
        if clustering_signal is not None:
            signals["narrative_clustering"] = clustering_signal

        # Need at least 2 signals
        if len(signals) < 2:
            return None

        # Get current narrative for market condition check
        current_narrative = None
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        # Dynamic weights based on market condition
        market_condition = self._get_market_condition(current_narrative["timestamp"])

        # Default weights
        weights = {
            "pattern": 0.20,
            "sentiment_event": 0.25,
            "price_momentum": 0.20,
            "sentiment_momentum": 0.15,
            "volatility_regime": 0.10,
            "narrative_clustering": 0.10,
        }

        # Adjust weights based on market condition
        if market_condition is not None:
            if market_condition > 0.5:
                # Bullish market: weight momentum signals higher
                weights["price_momentum"] = 0.30
                weights["sentiment_event"] = 0.20
                weights["pattern"] = 0.15
            elif market_condition < -0.5:
                # Bearish market: weight sentiment and volatility higher
                weights["sentiment_event"] = 0.35
                weights["volatility_regime"] = 0.15
                weights["price_momentum"] = 0.10

        # Calculate weighted score
        total_weight = sum(weights.get(k, 0.1) for k in signals.keys())
        weighted_sum = sum(signals[k] * weights.get(k, 0.1) for k in signals.keys())

        composite_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Apply volatility adjustment
        if "volatility_regime" in signals:
            # In high volatility, reduce position size
            composite_score *= 1 + signals["volatility_regime"] * 0.2

        return np.clip(composite_score, -1, 1)

    def _get_pattern_signal(self, narrative_id: str) -> Optional[float]:
        """
        Original pattern-based signal generation (normalized to -1 to 1).
        """
        # Get current narrative details
        narrative_timestamp = ""
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate current price position
        current_price_position = self.calculate_price_position(narrative_timestamp)
        if current_price_position is None:
            return None

        # Find similar historical narratives
        similar_narratives = self.search_similar_narratives(narrative_id)

        # Calculate correlation between price positions and returns
        price_positions = []
        price_returns = []

        for similar_narrative in similar_narratives:
            position = self.calculate_price_position(similar_narrative["timestamp"])
            if position is None:
                continue

            return_ = self.calculate_price_return(similar_narrative["timestamp"])
            if return_ is None:
                continue

            price_positions.append(position)
            price_returns.append(return_)

        # Need at least 2 data points for correlation
        if len(price_positions) < 2:
            return None

        # Calculate Pearson correlation
        corr_matrix = np.corrcoef(price_positions, price_returns)
        pearson_r = corr_matrix[0, 1]

        # Generate signal based on correlation and current price position
        if current_price_position > 0.5 and pearson_r > 0:
            return pearson_r  # Positive signal strength
        elif current_price_position < 0.5 and pearson_r < 0:
            return pearson_r  # Negative signal strength
        else:
            return 0.0

    def _get_sentiment_event_signal(self, narrative_id: str) -> Optional[float]:
        """
        Generate signal based on sentiment-event combination performance.
        """
        current_narrative = None
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        sentiment = current_narrative.get("sentiment", "neutral")
        event_type = current_narrative.get("event", "")

        if sentiment == "neutral" or not event_type:
            return 0.0

        # Find similar sentiment-event combinations
        similar_returns = []
        for narrative in self.narratives_stored:
            if narrative["timestamp"] >= current_narrative["timestamp"]:
                continue

            if (
                narrative.get("sentiment") == sentiment
                and narrative.get("event") == event_type
            ):

                ret = self.calculate_price_return(narrative["timestamp"])
                if ret is not None:
                    similar_returns.append(ret)

        if len(similar_returns) < 3:  # Need sufficient history
            return None

        # Calculate average return and consistency
        avg_return = np.mean(similar_returns)
        std_return = np.std(similar_returns)

        # Calculate signal strength (Sharpe-like ratio)
        if std_return > 0:
            signal_strength = avg_return / std_return
        else:
            signal_strength = np.sign(avg_return)

        # Adjust for sentiment direction
        if sentiment == "bullish":
            return np.clip(signal_strength, -1, 1)
        elif sentiment == "bearish":
            return np.clip(-signal_strength, -1, 1)

        return 0.0

    def _get_price_momentum_signal(self, narrative_id: str) -> Optional[float]:
        """
        Generate signal based on price momentum at narrative time.
        """
        narrative_timestamp = ""
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate short-term and long-term momentum
        short_window = min(24, self.config.look_back_period // 2)
        long_window = self.config.look_back_period

        # Short-term momentum
        short_start = (
            datetime.fromisoformat(narrative_timestamp) - timedelta(hours=short_window)
        ).isoformat()
        short_prices = []
        for ts, price in self.prices_stored:
            if short_start <= ts <= narrative_timestamp:
                short_prices.append(price)

        if len(short_prices) < 2:
            return None

        short_momentum = (short_prices[-1] - short_prices[0]) / short_prices[0]

        # Long-term momentum
        long_start = (
            datetime.fromisoformat(narrative_timestamp) - timedelta(hours=long_window)
        ).isoformat()
        long_prices = []
        for ts, price in self.prices_stored:
            if long_start <= ts <= narrative_timestamp:
                long_prices.append(price)

        if len(long_prices) < 2:
            return None

        long_momentum = (long_prices[-1] - long_prices[0]) / long_prices[0]

        # Generate signal: positive if short > long (momentum accelerating)
        momentum_diff = short_momentum - long_momentum

        # Normalize to -1 to 1 range
        return np.tanh(momentum_diff * 10)  # Scale factor of 10 for sensitivity

    def _get_sentiment_momentum_signal(self, narrative_id: str) -> Optional[float]:
        """
        Generate signal based on sentiment momentum (shift in narrative tone).
        """
        current_narrative = None
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        # Count sentiment in two windows: recent (6h) vs previous (6-12h)
        current_time = datetime.fromisoformat(current_narrative["timestamp"])
        recent_start = (current_time - timedelta(hours=6)).isoformat()
        previous_start = (current_time - timedelta(hours=12)).isoformat()

        recent_sentiment = {"bullish": 0, "bearish": 0, "neutral": 0}
        previous_sentiment = {"bullish": 0, "bearish": 0, "neutral": 0}

        for narrative in self.narratives_stored:
            ts = narrative["timestamp"]
            sentiment = narrative.get("sentiment", "neutral")

            if recent_start <= ts <= current_narrative["timestamp"]:
                recent_sentiment[sentiment] += 1
            elif previous_start <= ts < recent_start:
                previous_sentiment[sentiment] += 1

        recent_total = sum(recent_sentiment.values())
        previous_total = sum(previous_sentiment.values())

        if recent_total < 2 or previous_total < 2:
            return None

        # Calculate sentiment scores
        recent_score = (
            recent_sentiment["bullish"] - recent_sentiment["bearish"]
        ) / recent_total
        previous_score = (
            previous_sentiment["bullish"] - previous_sentiment["bearish"]
        ) / previous_total

        # Momentum is the change in sentiment
        sentiment_momentum = recent_score - previous_score

        # Align with current narrative sentiment
        current_sentiment = current_narrative.get("sentiment", "neutral")
        if current_sentiment == "bullish" and sentiment_momentum > 0:
            return sentiment_momentum  # Bullish momentum confirmed
        elif current_sentiment == "bearish" and sentiment_momentum < 0:
            return -sentiment_momentum  # Bearish momentum confirmed
        elif current_sentiment == "bullish" and sentiment_momentum < -0.3:
            return -0.5  # Strong contrarian signal
        elif current_sentiment == "bearish" and sentiment_momentum > 0.3:
            return 0.5  # Strong contrarian signal
        else:
            return 0.0

    def _get_market_condition(self, timestamp: str) -> Optional[float]:
        """
        Determine market condition: bullish (+1) or bearish (-1).
        Uses multiple timeframes and indicators.
        """
        # Calculate multiple moving averages
        ma_periods = [24, 72, 168]  # 1 day, 3 days, 1 week
        ma_values = []

        for period in ma_periods:
            start_time = (
                datetime.fromisoformat(timestamp) - timedelta(hours=period)
            ).isoformat()

            prices = []
            for ts, price in self.prices_stored:
                if start_time <= ts <= timestamp:
                    prices.append(price)

            if len(prices) > period // 2:
                ma_values.append(np.mean(prices))
            else:
                return None

        if len(ma_values) < len(ma_periods):
            return None

        current_price = self.get_price_at_timestamp(timestamp)
        if current_price is None:
            return None

        # Market condition score
        score = 0.0

        # Price vs MAs
        for i, ma in enumerate(ma_values):
            weight = 1 / (i + 1)  # Shorter MAs have higher weight
            if current_price > ma:
                score += weight
            else:
                score -= weight

        # MA alignment (shorter > longer = bullish)
        if len(ma_values) >= 2:
            for i in range(len(ma_values) - 1):
                if ma_values[i] > ma_values[i + 1]:
                    score += 0.3
                else:
                    score -= 0.3

        return np.tanh(score)  # Normalize to [-1, 1]

    def _get_volatility_regime_signal(self, narrative_id: str) -> Optional[float]:
        """
        Generate signal based on volatility regime.
        High volatility + bearish = stronger bearish signal
        Low volatility + trend following = stronger signal
        """
        narrative_timestamp = ""
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate realized volatility
        vol_window = min(48, self.config.look_back_period)
        start_time = (
            datetime.fromisoformat(narrative_timestamp) - timedelta(hours=vol_window)
        ).isoformat()

        prices = []
        for ts, price in self.prices_stored:
            if start_time <= ts <= narrative_timestamp:
                prices.append(price)

        if len(prices) < 10:
            return None

        # Calculate hourly returns
        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns) * np.sqrt(24)  # Annualized

        # Calculate historical average volatility (simplified version)
        # Use percentile approach
        if current_vol > 0.5:  # High volatility threshold
            return -0.3  # Reduce position confidence
        elif current_vol < 0.2:  # Low volatility threshold
            return 0.3  # Increase confidence
        else:
            return 0.0

    def _get_narrative_clustering_signal(self, narrative_id: str) -> Optional[float]:
        """
        Advanced signal based on narrative clustering and regime detection.
        Groups narratives into clusters and identifies regime-specific patterns.
        """
        current_narrative = None
        for narrative in self.narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        # Define narrative "fingerprint" based on keywords, sentiment, and event
        current_keywords = set(current_narrative.get("pattern_keywords", []))
        current_sentiment = current_narrative.get("sentiment", "neutral")
        current_event = current_narrative.get("event", "")

        # Find narratives with similar fingerprints
        similar_narratives = []
        for narrative in self.narratives_stored:
            if narrative["timestamp"] >= current_narrative["timestamp"]:
                continue

            # Calculate similarity score
            keywords_overlap = len(
                current_keywords.intersection(
                    set(narrative.get("pattern_keywords", []))
                )
            )

            sentiment_match = (
                1 if narrative.get("sentiment") == current_sentiment else 0
            )
            event_match = 1 if narrative.get("event") == current_event else 0

            similarity = (
                keywords_overlap * 0.5 + sentiment_match * 0.3 + event_match * 0.2
            )

            if similarity > 1.5:  # Threshold for similarity
                similar_narratives.append(narrative)

        if len(similar_narratives) < 3:
            return None

        # Analyze performance in different market conditions
        bullish_market_returns = []
        bearish_market_returns = []

        for narrative in similar_narratives:
            # Determine market condition at narrative time
            market_condition = self._get_market_condition(narrative["timestamp"])
            ret = self.calculate_price_return(narrative["timestamp"])

            if ret is not None and market_condition is not None:
                if market_condition > 0:
                    bullish_market_returns.append(ret)
                else:
                    bearish_market_returns.append(ret)

        # Get current market condition
        current_market = self._get_market_condition(current_narrative["timestamp"])
        if current_market is None:
            return None

        # Generate signal based on historical performance in similar market conditions
        if current_market > 0 and len(bullish_market_returns) >= 2:
            avg_return = np.mean(bullish_market_returns)
            consistency = 1 / (np.std(bullish_market_returns) + 0.01)
            return np.tanh(avg_return * consistency * 5)
        elif current_market < 0 and len(bearish_market_returns) >= 2:
            avg_return = np.mean(bearish_market_returns)
            consistency = 1 / (np.std(bearish_market_returns) + 0.01)
            return np.tanh(avg_return * consistency * 5)

        return 0.0

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
            annualized_vol = period_std * np.sqrt(365 * 24 / self.config.hold_period)
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
