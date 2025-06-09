"""
Example backtest script for NarrativeAgent with enhanced transaction costs.
"""

import os
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

from src.narrative_agent import NarrativeAgent, NarrativeAgentConfig


def run_backtest():
    """
    Run a backtest of a NarrativeAgent with enhanced transaction cost modeling.
    """

    api_key = os.getenv("SENTICHAIN_API_KEY")

    if not api_key:
        print("ERROR: SENTICHAIN_API_KEY environment variable not set!")
        print("\nPlease set your API key using one of these methods:")
        print("\nWindows:")
        print("  set SENTICHAIN_API_KEY=your-api-key-here")
        print("\nLinux/Mac:")
        print("  export SENTICHAIN_API_KEY=your-api-key-here")
        print("\nOr in Python:")
        print("  os.environ['SENTICHAIN_API_KEY'] = 'your-api-key-here'")
        return

    print(f"Using API key from environment variable (length: {len(api_key)})")

    # Check for cache control
    clear_cache = os.getenv("CLEAR_CACHE", "false").lower() == "true"
    use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
    cache_dir = os.getenv("CACHE_DIR", ".narrative_cache")

    config = NarrativeAgentConfig(
        ticker="BTC",
        look_back_period=6,
        hold_period=1,
        count_common_threshold=5,
        transaction_cost=10,
        use_enhanced_costs=True,
        gas_fee_usd=50.0,
        amm_liquidity_usd=5_000_000.0,
        position_size_usd=50_000.0,
    )

    # Create agent with caching support
    agent = NarrativeAgent(config, api_key, use_cache=use_cache, cache_dir=cache_dir)

    # Clear cache if requested
    if clear_cache and use_cache:
        print("\nClearing cache...")
        agent.clear_cache()

    # Show cache
    if use_cache:
        cache_info = agent.get_cache_info()
        print("\nCache info:")
        print(f"  - Directory: {cache_info['cache_dir']}")
        print(f"  - Narrative files: {cache_info['narrative_files']}")
        print(f"  - Price files: {cache_info['price_files']}")
        print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")

    # Backtest parameters
    start_date = os.getenv("BACKTEST_START_DATE", "2025-02-01T00:00:00")
    num_days = int(os.getenv("BACKTEST_DAYS", "30"))

    # Run backtest
    print(f"\nRunning backtest from {start_date} for {num_days} days...")
    print("=" * 60)

    for day in range(num_days):
        timestamp = (
            datetime.fromisoformat(start_date) + timedelta(days=day)
        ).isoformat()

        print(f"\nDay {day + 1}/{num_days}: {timestamp[:10]}")
        agent.update(timestamp)

    # Finalize any remaining open positions
    final_timestamp = (
        datetime.fromisoformat(start_date) + timedelta(days=num_days)
    ).isoformat()
    agent.finalize_positions(final_timestamp)

    print("\nBacktest complete.")

    # Show cache after backtest
    if use_cache:
        cache_info = agent.get_cache_info()
        print("\nCache info after backtest:")
        print(f"  - Narrative files: {cache_info['narrative_files']}")
        print(f"  - Price files: {cache_info['price_files']}")
        print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")

    # Get performance metrics
    df = agent.get_performance_dataframe()

    if df.empty:
        print("\nNo positions were taken during the backtest period.")
        return

    # Print performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total positions: {len(df)}")
    print(f"Total return: {df['cum_return'].iloc[-1]:.2%}")
    print(f"Max drawdown: {df['max_drawdown'].iloc[-1]:.2%}")
    print(f"Annualized volatility: {df['vol_annualized'].iloc[-1]:.2%}")

    # Calculate win rate
    wins = (df["position_return"] > 0).sum()
    total = len(df)
    print(f"Win rate: {wins}/{total} ({wins/total:.1%})")

    # Calculate average return per position
    avg_return = df["position_return"].mean()
    print(f"Average return per position: {avg_return:.2%}")

    # Print transaction cost summary if using enhanced model
    if config.use_enhanced_costs:
        print("\n" + "=" * 60)
        print("TRANSACTION COST ANALYSIS")
        print("=" * 60)

        cost_summary = agent.get_transaction_cost_summary()
        if "transaction_count" in cost_summary:
            print(f"Total transactions: {cost_summary['transaction_count']}")
            print(f"Total gas fees: ${cost_summary['total_gas_paid']:,.2f}")
            print(f"Total slippage costs: ${cost_summary['total_slippage_paid']:,.2f}")
            print(f"Total protocol fees: ${cost_summary['total_fees_paid']:,.2f}")
            print(f"Total transaction costs: ${cost_summary['total_costs']:,.2f}")
            print("\nAverage per transaction:")
            print(f"  Gas: ${cost_summary['avg_gas_per_tx']:.2f}")
            print(f"  Slippage: ${cost_summary['avg_slippage_per_tx']:.2f}")
            print(f"  Total: ${cost_summary['avg_total_cost_per_tx']:.2f}")
            print("\nCost breakdown:")
            print(f"  Gas: {cost_summary['gas_pct_of_total']:.1f}%")
            print(f"  Slippage: {cost_summary['slippage_pct_of_total']:.1f}%")
            print(f"  Protocol fees: {cost_summary['fees_pct_of_total']:.1f}%")

            # Calculate cost impact on returns
            if "total_cost_usd" in df.columns:
                total_position_value = config.position_size_usd * len(df)
                cost_impact_pct = (
                    cost_summary["total_costs"] / total_position_value
                ) * 100
                print(f"\nTotal cost impact: {cost_impact_pct:.2f}% of capital traded")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Cumulative returns
    ax1 = axes[0, 0]
    ax1.plot(df.index, df["cum_return"], linewidth=2)
    ax1.set_title(f"Cumulative Returns - {config.ticker}", fontsize=14)
    ax1.set_xlabel("Position Number")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # 2. Per-position returns
    ax2 = axes[0, 1]
    colors = ["green" if r > 0 else "red" for r in df["position_return"]]
    ax2.bar(df.index, df["position_return"], color=colors, alpha=0.6)
    ax2.set_title("Returns per Position", fontsize=14)
    ax2.set_xlabel("Position Number")
    ax2.set_ylabel("Return")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # 3. Transaction costs per position (if available)
    ax3 = axes[1, 0]
    if "total_cost_usd" in df.columns:
        ax3.bar(df.index, df["total_cost_usd"], color="orange", alpha=0.6)
        ax3.set_title("Transaction Costs per Position", fontsize=14)
        ax3.set_xlabel("Position Number")
        ax3.set_ylabel("Cost (USD)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Transaction cost data not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Transaction Costs per Position", fontsize=14)

    # 4. Slippage analysis (if available)
    ax4 = axes[1, 1]
    if "entry_slippage_bps" in df.columns and "exit_slippage_bps" in df.columns:
        positions = df.index
        width = 0.35
        ax4.bar(
            positions - width / 2,
            df["entry_slippage_bps"],
            width,
            label="Entry slippage",
            alpha=0.6,
        )
        ax4.bar(
            positions + width / 2,
            df["exit_slippage_bps"],
            width,
            label="Exit slippage",
            alpha=0.6,
        )
        ax4.set_title("Slippage per Position", fontsize=14)
        ax4.set_xlabel("Position Number")
        ax4.set_ylabel("Slippage (bps)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "Slippage data not available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Slippage per Position", fontsize=14)

    plt.tight_layout()
    plt.show()

    # Save performance data with enhanced metrics
    df.to_csv(f"backtest_results_{config.ticker}_enhanced.csv", index=False)
    print(f"\nPerformance data saved to backtest_results_{config.ticker}_enhanced.csv")


if __name__ == "__main__":
    run_backtest()
