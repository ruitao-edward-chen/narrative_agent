"""
Strategy module for the Narrative Agent.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod


class SignalGenerator(ABC):
    """
    Base class for signal generators.
    """

    @abstractmethod
    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate a signal for the given narrative.
        """
        pass


class PatternSignal(SignalGenerator):
    """
    Pattern-based signal generation.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on historical pattern correlation.
        """
        narratives_stored = context["narratives_stored"]
        calculate_price_position = context["calculate_price_position"]
        calculate_price_return = context["calculate_price_return"]
        search_similar_narratives = context["search_similar_narratives"]

        # Get current narrative details
        narrative_timestamp = ""
        for narrative in narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate current price position
        current_price_position = calculate_price_position(narrative_timestamp)
        if current_price_position is None:
            return None

        # Find similar historical narratives
        similar_narratives = search_similar_narratives(narrative_id)

        # Calculate correlation between price positions and returns
        price_positions = []
        price_returns = []

        for similar_narrative in similar_narratives:
            position = calculate_price_position(similar_narrative["timestamp"])
            if position is None:
                continue

            return_ = calculate_price_return(similar_narrative["timestamp"])
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


class SentimentEventSignal(SignalGenerator):
    """
    Signal based on sentiment-event combination performance.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on sentiment-event combinations.
        """
        narratives_stored = context["narratives_stored"]
        calculate_price_return = context["calculate_price_return"]

        current_narrative = None
        for narrative in narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        sentiment = current_narrative.get("sentiment", "neutral").lower().strip()
        # Handle typos
        if sentiment in ["bullish", "bulish"]:
            sentiment = "bullish"
        elif sentiment == "bearish":
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        event_type = current_narrative.get("event", "")

        if sentiment == "neutral" or not event_type:
            return 0.0

        # Find similar sentiment-event combinations
        similar_returns = []
        for narrative in narratives_stored:
            if narrative["timestamp"] >= current_narrative["timestamp"]:
                continue

            narrative_sentiment = narrative.get("sentiment", "neutral").lower().strip()
            # Handle typos in historical narratives
            if narrative_sentiment in ["bullish", "bulish"]:
                narrative_sentiment = "bullish"
            elif narrative_sentiment == "bearish":
                narrative_sentiment = "bearish"
            else:
                narrative_sentiment = "neutral"

            if (
                narrative_sentiment == sentiment
                and narrative.get("event") == event_type
            ):
                ret = calculate_price_return(narrative["timestamp"])
                if ret is not None:
                    similar_returns.append(ret)

        # Need sufficient history
        if len(similar_returns) < 3:
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


class PriceMomentumSignal(SignalGenerator):
    """
    Signal based on price momentum.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on price momentum at narrative time.
        """
        narratives_stored = context["narratives_stored"]
        prices_stored = context["prices_stored"]
        config = context["config"]

        narrative_timestamp = ""
        for narrative in narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate short-term and long-term momentum
        short_window = min(24, config.look_back_period // 2)
        long_window = config.look_back_period

        # Short-term momentum
        short_start = (
            datetime.fromisoformat(narrative_timestamp) - timedelta(hours=short_window)
        ).isoformat()
        short_prices = []
        for ts, price in prices_stored:
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
        for ts, price in prices_stored:
            if long_start <= ts <= narrative_timestamp:
                long_prices.append(price)

        if len(long_prices) < 2:
            return None

        long_momentum = (long_prices[-1] - long_prices[0]) / long_prices[0]

        # Generate signal: positive if short > long (momentum accelerating)
        momentum_diff = short_momentum - long_momentum

        # Normalize to -1 to 1 range
        return np.tanh(momentum_diff * 10)  # Scale factor of 10 for sensitivity


class SentimentMomentumSignal(SignalGenerator):
    """
    Signal based on sentiment momentum.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on sentiment momentum (shift in narrative tone).
        """
        narratives_stored = context["narratives_stored"]

        current_narrative = None
        for narrative in narratives_stored:
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

        for narrative in narratives_stored:
            ts = narrative["timestamp"]
            sentiment = narrative.get("sentiment", "neutral")

            # Handle typos and normalize sentiment
            sentiment = sentiment.lower().strip()
            if sentiment in ["bullish", "bulish"]:  # Handle typo
                sentiment = "bullish"
            elif sentiment == "bearish":
                sentiment = "bearish"
            else:
                sentiment = "neutral"

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
        current_sentiment = (
            current_narrative.get("sentiment", "neutral").lower().strip()
        )
        # Handle typo for current sentiment too
        if current_sentiment in ["bullish", "bulish"]:
            current_sentiment = "bullish"

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


class VolatilityRegimeSignal(SignalGenerator):
    """
    Signal based on volatility regime.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on volatility regime.
        """
        narratives_stored = context["narratives_stored"]
        prices_stored = context["prices_stored"]
        config = context["config"]

        narrative_timestamp = ""
        for narrative in narratives_stored:
            if narrative["ID"] == narrative_id:
                narrative_timestamp = narrative.get("timestamp", "")
                break

        if not narrative_timestamp:
            return None

        # Calculate realized volatility
        vol_window = min(48, config.look_back_period)
        start_time = (
            datetime.fromisoformat(narrative_timestamp) - timedelta(hours=vol_window)
        ).isoformat()

        prices = []
        for ts, price in prices_stored:
            if start_time <= ts <= narrative_timestamp:
                prices.append(price)

        if len(prices) < 10:
            return None

        # Calculate hourly returns
        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns) * np.sqrt(24)  # Annualized

        # Use percentile approach
        if current_vol > 0.5:  # High volatility threshold
            return -0.3  # Reduce position confidence
        elif current_vol < 0.2:  # Low volatility threshold
            return 0.3  # Increase confidence
        else:
            return 0.0


class NarrativeClusteringSignal(SignalGenerator):
    """
    Signal based on narrative clustering and regime detection.
    """

    def generate_signal(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Generate signal based on narrative clustering.
        """
        narratives_stored = context["narratives_stored"]
        calculate_price_return = context["calculate_price_return"]
        get_market_condition = context["get_market_condition"]

        current_narrative = None
        for narrative in narratives_stored:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        # Define narrative "fingerprint"
        current_keywords = set(current_narrative.get("pattern_keywords", []))
        current_sentiment = (
            current_narrative.get("sentiment", "neutral").lower().strip()
        )
        # Handle typos
        if current_sentiment in ["bullish", "bulish"]:
            current_sentiment = "bullish"
        elif current_sentiment == "bearish":
            current_sentiment = "bearish"
        else:
            current_sentiment = "neutral"

        current_event = current_narrative.get("event", "")

        # Find narratives with similar fingerprints
        similar_narratives = []
        for narrative in narratives_stored:
            if narrative["timestamp"] >= current_narrative["timestamp"]:
                continue

            # Calculate similarity score
            keywords_overlap = len(
                current_keywords.intersection(
                    set(narrative.get("pattern_keywords", []))
                )
            )

            narrative_sentiment = narrative.get("sentiment", "neutral").lower().strip()
            # Handle typos
            if narrative_sentiment in ["bullish", "bulish"]:
                narrative_sentiment = "bullish"
            elif narrative_sentiment == "bearish":
                narrative_sentiment = "bearish"
            else:
                narrative_sentiment = "neutral"

            sentiment_match = 1 if narrative_sentiment == current_sentiment else 0
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
            market_condition = get_market_condition(narrative["timestamp"])
            ret = calculate_price_return(narrative["timestamp"])

            if ret is not None and market_condition is not None:
                if market_condition > 0:
                    bullish_market_returns.append(ret)
                else:
                    bearish_market_returns.append(ret)

        # Get current market condition
        current_market = get_market_condition(current_narrative["timestamp"])
        if current_market is None:
            return None

        # Generate signal based on historical performance
        if current_market > 0 and len(bullish_market_returns) >= 2:
            avg_return = np.mean(bullish_market_returns)
            consistency = 1 / (np.std(bullish_market_returns) + 0.01)
            return np.tanh(avg_return * consistency * 5)
        elif current_market < 0 and len(bearish_market_returns) >= 2:
            avg_return = np.mean(bearish_market_returns)
            consistency = 1 / (np.std(bearish_market_returns) + 0.01)
            return np.tanh(avg_return * consistency * 5)

        return 0.0


class CompositeStrategy:
    """
    Manages multiple signal generators and combines their outputs.
    """

    def __init__(self):
        """
        Initialize the composite strategy with signal generators.
        """
        self.signal_generators = {
            "pattern": PatternSignal(),
            "sentiment_event": SentimentEventSignal(),
            "price_momentum": PriceMomentumSignal(),
            "sentiment_momentum": SentimentMomentumSignal(),
            "volatility_regime": VolatilityRegimeSignal(),
            "narrative_clustering": NarrativeClusteringSignal(),
        }

        # Default weights
        self.default_weights = {
            "pattern": 0.20,
            "sentiment_event": 0.25,
            "price_momentum": 0.20,
            "sentiment_momentum": 0.15,
            "volatility_regime": 0.10,
            "narrative_clustering": 0.10,
        }

    def calculate_composite_score(
        self, narrative_id: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate composite score from all signals.
        """
        # Get all signals
        signals = {}
        for name, generator in self.signal_generators.items():
            signal = generator.generate_signal(narrative_id, context)
            if signal is not None:
                signals[name] = signal

        # Need at least 2 signals
        if len(signals) < 2:
            return None

        # Get current narrative for market condition check
        current_narrative = None
        for narrative in context["narratives_stored"]:
            if narrative["ID"] == narrative_id:
                current_narrative = narrative
                break

        if not current_narrative:
            return None

        # Dynamic weights based on market condition
        market_condition = context["get_market_condition"](
            current_narrative["timestamp"]
        )
        weights = self._get_dynamic_weights(market_condition)

        # Calculate weighted score
        total_weight = sum(weights.get(k, 0.1) for k in signals.keys())
        weighted_sum = sum(signals[k] * weights.get(k, 0.1) for k in signals.keys())

        composite_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Apply volatility adjustment
        if "volatility_regime" in signals:
            composite_score *= 1 + signals["volatility_regime"] * 0.2

        return np.clip(composite_score, -1, 1)

    def _get_dynamic_weights(
        self, market_condition: Optional[float]
    ) -> Dict[str, float]:
        """
        Get dynamic weights based on market condition.
        """
        weights = self.default_weights.copy()

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

        return weights


def get_market_condition(
    timestamp: str, prices_stored: List[Tuple[str, float]], get_price_at_timestamp
) -> Optional[float]:
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
        for ts, price in prices_stored:
            if start_time <= ts <= timestamp:
                prices.append(price)

        if len(prices) > period // 2:
            ma_values.append(np.mean(prices))
        else:
            return None

    if len(ma_values) < len(ma_periods):
        return None

    current_price = get_price_at_timestamp(timestamp)
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
