"""
Test AMM Pool Simulation and Transaction Cost Calculations.
"""

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.narrative_agent import AMMPool, TransactionCostModel
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_amm_mechanics():
    """
    Show basic AMM pool mechanics.
    """
    print("=" * 60)
    print("AMM POOL MECHANICS DEMONSTRATION")
    print("=" * 60)

    # Create a pool with 1M liquidity
    pool = AMMPool(initial_liquidity_usd=1_000_000, fee_tier=0.003)

    # Sync to BTC price of 50,000
    btc_price = 50_000
    pool.sync_to_price(btc_price)

    print(f"\nPool initialized with ${pool.initial_liquidity:,.0f} liquidity")
    print(f"BTC price: ${btc_price:,.2f}")
    print("Pool reserves:")
    print(f"  BTC: {pool.reserve_base:.4f}")
    print(f"  USD: ${pool.reserve_quote:,.2f}")
    print(f"  k (constant): {pool.k:,.2f}")

    # Test different trade sizes
    trade_sizes_usd = [1_000, 10_000, 50_000, 100_000, 200_000]

    print("\n" + "-" * 60)
    print("SLIPPAGE ANALYSIS FOR DIFFERENT TRADE SIZES")
    print("-" * 60)
    print(
        f"{'Trade Size':>12} | {'Slippage (bps)':>14} | {'Effective Price':>15} | {'Price Impact %':>14}"
    )
    print("-" * 60)

    for size in trade_sizes_usd:
        slippage_bps = pool.calculate_slippage_bps(size, btc_price, is_buy=True)

        # Calculate actual swap to get more details
        btc_amount = size / btc_price
        amount_out, price_impact, effective_price = pool.calculate_swap_amounts(
            btc_amount, is_base_to_quote=False
        )

        print(
            f"${size:>11,} | {slippage_bps:>14.1f} | ${effective_price:>14,.2f} | {price_impact*100:>14.2f}%"
        )


def analyze_liquidity_impact():
    """
    Analyze how pool liquidity affects slippage.
    """
    print("\n" * 2)
    print("=" * 60)
    print("LIQUIDITY IMPACT ANALYSIS")
    print("=" * 60)

    liquidity_levels = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    trade_size = 50_000  # $50k trade
    btc_price = 50_000

    slippages = []

    print(f"\nFixed trade size: ${trade_size:,}")
    print(f"BTC price: ${btc_price:,}")
    print("\n" + "-" * 40)
    print(f"{'Pool Liquidity':>15} | {'Slippage (bps)':>15}")
    print("-" * 40)

    for liquidity in liquidity_levels:
        pool = AMMPool(initial_liquidity_usd=liquidity)
        pool.sync_to_price(btc_price)
        slippage_bps = pool.calculate_slippage_bps(trade_size, btc_price, is_buy=True)
        slippages.append(slippage_bps)
        print(f"${liquidity:>14,} | {slippage_bps:>15.1f}")

    # Plot liquidity vs slippage
    plt.figure(figsize=(10, 6))
    plt.plot(liquidity_levels, slippages, "b-o", linewidth=2, markersize=8)
    plt.xlabel("Pool Liquidity (USD)", fontsize=12)
    plt.ylabel("Slippage (basis points)", fontsize=12)
    plt.title(f"Slippage vs Pool Liquidity for ${trade_size:,} Trade", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

    plt.tight_layout()
    plt.savefig("liquidity_impact.png", dpi=150)
    print("\nChart saved as 'liquidity_impact.png'")


def demonstrate_transaction_costs():
    """
    Show complete transaction cost breakdown.
    """
    print("\n" * 2)
    print("=" * 60)
    print("TRANSACTION COST MODEL DEMONSTRATION")
    print("=" * 60)

    # Create transaction cost model
    cost_model = TransactionCostModel(
        gas_fee_usd=50.0, amm_liquidity_usd=5_000_000, position_size_usd=50_000
    )

    btc_price = 50_000

    # Calculate costs for a long position
    print(
        f"\nPosition: LONG ${cost_model.position_size_usd:,} of BTC at ${btc_price:,}"
    )

    # Entry costs
    entry_costs = cost_model.calculate_transaction_costs(
        cost_model.position_size_usd, btc_price, is_buy=True, is_entry=True
    )

    print("\nEntry Costs:")
    print(f"  Gas fee: ${entry_costs.gas_fee_usd:.2f}")
    print(
        f"  Slippage: {entry_costs.slippage_bps:.1f} bps (${entry_costs.slippage_usd:.2f})"
    )
    print(f"  Protocol fee: ${entry_costs.protocol_fee_usd:.2f}")
    print(
        f"  Total: ${entry_costs.total_cost_usd:.2f} ({entry_costs.total_cost_bps:.1f} bps)"
    )
    print(f"  Effective entry price: ${entry_costs.effective_price:,.2f}")

    # Simulate price movement
    exit_price = btc_price * 1.02

    # Exit costs
    exit_costs = cost_model.calculate_transaction_costs(
        cost_model.position_size_usd, exit_price, is_buy=False, is_entry=False
    )

    print(f"\nExit Costs (at ${exit_price:,}):")
    print(f"  Gas fee: ${exit_costs.gas_fee_usd:.2f}")
    print(
        f"  Slippage: {exit_costs.slippage_bps:.1f} bps (${exit_costs.slippage_usd:.2f})"
    )
    print(f"  Protocol fee: ${exit_costs.protocol_fee_usd:.2f}")
    print(
        f"  Total: ${exit_costs.total_cost_usd:.2f} ({exit_costs.total_cost_bps:.1f} bps)"
    )

    # Round trip analysis
    round_trip = cost_model.calculate_round_trip_costs(
        cost_model.position_size_usd, btc_price, exit_price, position_type=1
    )

    print("\nRound Trip Summary:")
    print(f"  Gross P&L: ${round_trip['gross_pnl']:,.2f}")
    print(
        f"  Total costs: ${round_trip['total_cost_usd']:,.2f} ({round_trip['total_cost_bps']:.1f} bps)"
    )
    print(f"  Net P&L: ${round_trip['net_pnl']:,.2f}")
    print(f"  Costs as % of gross: {round_trip['cost_as_pct_of_gross']:.1f}%")

    # Break-even analysis
    breakeven = cost_model.estimate_break_even_move(
        cost_model.position_size_usd, btc_price, position_type=1
    )
    print(f"\nBreak-even price move required: {breakeven:.2f}%")


def analyze_trade_size_impact():
    """
    Analyze how trade size affects costs.
    """
    print("\n" * 2)
    print("=" * 60)
    print("TRADE SIZE IMPACT ANALYSIS")
    print("=" * 60)

    # Fixed parameters
    gas_fee = 50.0
    liquidity = 5_000_000
    btc_price = 50_000

    trade_sizes = np.linspace(1_000, 200_000, 50)
    gas_costs_pct = []
    slippage_costs_pct = []
    total_costs_pct = []

    for size in trade_sizes:
        cost_model = TransactionCostModel(
            gas_fee_usd=gas_fee, amm_liquidity_usd=liquidity, position_size_usd=size
        )

        costs = cost_model.calculate_transaction_costs(size, btc_price, is_buy=True)

        gas_pct = (costs.gas_fee_usd / size) * 100
        slippage_pct = (costs.slippage_usd / size) * 100
        total_pct = (costs.total_cost_usd / size) * 100

        gas_costs_pct.append(gas_pct)
        slippage_costs_pct.append(slippage_pct)
        total_costs_pct.append(total_pct)

    # Create visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(trade_sizes / 1000, gas_costs_pct, "g-", label="Gas", linewidth=2)
    plt.plot(
        trade_sizes / 1000, slippage_costs_pct, "b-", label="Slippage", linewidth=2
    )
    plt.plot(trade_sizes / 1000, total_costs_pct, "r-", label="Total", linewidth=2)
    plt.xlabel("Trade Size ($K)", fontsize=12)
    plt.ylabel("Cost as % of Trade", fontsize=12)
    plt.title("Transaction Costs vs Trade Size", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.stackplot(
        trade_sizes / 1000,
        gas_costs_pct,
        slippage_costs_pct,
        labels=["Gas", "Slippage"],
        colors=["green", "blue"],
        alpha=0.7,
    )
    plt.xlabel("Trade Size ($K)", fontsize=12)
    plt.ylabel("Cost as % of Trade", fontsize=12)
    plt.title("Cost Composition by Trade Size", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trade_size_impact.png", dpi=150)
    print("\nChart saved as 'trade_size_impact.png'")

    # Print some key insights
    print("\nKey Insights:")
    print("- For small trades (<$10k), gas dominates costs")
    print("- For large trades (>$100k), slippage dominates")
    print("- Optimal trade size for this liquidity: ~$20-50k")


if __name__ == "__main__":
    demonstrate_amm_mechanics()
    analyze_liquidity_impact()
    demonstrate_transaction_costs()
    analyze_trade_size_impact()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThis demonstrates how our transaction cost model provides")
    print("realistic estimates for DeFi trading, accounting for both")
    print("fixed costs (gas) and variable costs (slippage).")

    plt.show()
