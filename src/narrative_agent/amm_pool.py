"""
AMM Pool Simulator for DeFi transaction cost modeling.
"""

from typing import Tuple, Optional


class AMMPool:
    """
    Simulates a constant product AMM pool (like Uniswap V2) for slippage calculation.
    """

    def __init__(self, initial_liquidity_usd: float, fee_tier: float = 0.003):
        """
        Initialize AMM pool with USD liquidity value.
        """
        self.fee_tier = fee_tier
        self.initial_liquidity = initial_liquidity_usd
        self.reserve_base: Optional[float] = None
        self.reserve_quote: Optional[float] = None
        self.k: Optional[float] = None

    def sync_to_price(self, external_price: float) -> None:
        """
        Sync pool reserves to match external price while maintaining TVL.
        Simulates arbitrageurs rebalancing the pool.
        """
        self.reserve_quote = self.initial_liquidity / 2
        self.reserve_base = self.reserve_quote / external_price
        self.k = self.reserve_base * self.reserve_quote

    def calculate_swap_amounts(
        self, amount_in: float, is_base_to_quote: bool
    ) -> Tuple[float, float, float]:
        """
        Calculate output amount and slippage for a swap.
        """
        if self.reserve_base is None or self.reserve_quote is None:
            raise ValueError("Pool not synced to price. Call sync_to_price first.")

        # Apply fee to input
        amount_in_with_fee = amount_in * (1 - self.fee_tier)

        if is_base_to_quote:
            # Swapping base (e.g., BTC) for quote (USD)
            new_reserve_base = self.reserve_base + amount_in_with_fee
            new_reserve_quote = self.k / new_reserve_base
            amount_out = self.reserve_quote - new_reserve_quote

            # Price calculations
            initial_price = self.reserve_quote / self.reserve_base
            final_price = new_reserve_quote / new_reserve_base
            # Fixed: Use amount_in_with_fee to exclude trading fee from slippage
            effective_price = amount_out / amount_in_with_fee

        else:
            # Swapping quote (USD) for base (e.g. BTC)
            new_reserve_quote = self.reserve_quote + amount_in_with_fee
            new_reserve_base = self.k / new_reserve_quote
            amount_out = self.reserve_base - new_reserve_base

            # Price calculations
            initial_price = self.reserve_quote / self.reserve_base
            final_price = new_reserve_quote / new_reserve_base
            # Fixed: Use amount_in_with_fee to exclude trading fee from slippage
            effective_price = amount_in_with_fee / amount_out

        # Calculate price impact as percentage change
        price_impact = abs(final_price - initial_price) / initial_price

        return amount_out, price_impact, effective_price

    def calculate_slippage_bps(
        self, position_value_usd: float, current_price: float, is_buy: bool
    ) -> float:
        """
        Calculate slippage in basis points for a position.
        """
        if is_buy:
            # Buying base asset with USD
            amount_in = position_value_usd
            _, price_impact, effective_price = self.calculate_swap_amounts(
                amount_in, is_base_to_quote=False
            )
            slippage = (effective_price - current_price) / current_price
        else:
            # Selling base asset for USD
            amount_in = position_value_usd / current_price
            _, price_impact, effective_price = self.calculate_swap_amounts(
                amount_in, is_base_to_quote=True
            )
            slippage = (current_price - effective_price) / current_price

        # Convert to bps
        return slippage * 10000

    def get_pool_metrics(self, current_price: float) -> dict:
        """
        Get current pool metrics for monitoring.
        """
        if self.reserve_base is None:
            return {"error": "Pool not initialized"}

        pool_price = self.reserve_quote / self.reserve_base
        price_deviation = (pool_price - current_price) / current_price

        return {
            "reserve_base": self.reserve_base,
            "reserve_quote": self.reserve_quote,
            "pool_price": pool_price,
            "external_price": current_price,
            "price_deviation_pct": price_deviation * 100,
            "tvl_usd": self.reserve_base * current_price + self.reserve_quote,
        }
