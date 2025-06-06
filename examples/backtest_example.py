"""
Example backtest script for NarrativeAgent.
"""

from datetime import datetime, timedelta
import joblib
from matplotlib import pyplot as plt

from src.narrative_agent import NarrativeAgent, NarrativeAgentConfig


def run_backtest():
    """
    Run a backtest of a NarrativeAgent.
    """
    config = NarrativeAgentConfig(
        ticker="BTC",
        look_back_period=6,
        hold_period=1,
        transaction_cost=1,
        count_common_threshold=5,
    )

    api_key = "<better-to-be-env-variable>"

    # Try to load existing agent or create a new one.
    agent_filename = f"{config.ticker}_{config.look_back_period}_{config.hold_period}_{config.transaction_cost}.pkl"

    try:
        agent = joblib.load(agent_filename)
        print(f"Loaded existing agent from {agent_filename}")
    except FileNotFoundError:
        agent = NarrativeAgent(config, api_key)
        print("Created new agent")

    # Backtest parameters
    start_date = "2025-02-01T00:00:00"
    num_days = 30

    # Run backtest
    print(f"\nRunning backtest from {start_date} for {num_days} days...")
    print("=" * 60)

    for day in range(num_days):
        timestamp = (
            datetime.fromisoformat(start_date) + timedelta(days=day)
        ).isoformat()

        print(f"\nDay {day + 1}/{num_days}: {timestamp[:10]}")
        agent.update(timestamp)

        # Save agent state periodically
        if (day + 1) % 10 == 0:
            joblib.dump(agent, agent_filename)
            print(f"Saved agent state to {agent_filename}")

    # Finalize any remaining open positions
    final_timestamp = (
        datetime.fromisoformat(start_date) + timedelta(days=num_days)
    ).isoformat()
    agent.finalize_positions(final_timestamp)

    # Save final state
    joblib.dump(agent, agent_filename)
    print(f"\nBacktest complete. Final state saved to {agent_filename}")

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

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cum_return"], linewidth=2)
    plt.title(f"Cumulative Returns - {config.ticker}", fontsize=16)
    plt.xlabel("Position Number", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Format y-axis as percentage
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.show()

    # Save performance data
    df.to_csv(f"backtest_results_{config.ticker}.csv", index=False)
    print(f"\nPerformance data saved to backtest_results_{config.ticker}.csv")


if __name__ == "__main__":
    run_backtest()
