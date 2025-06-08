# Narrative Agent

An algorithmic trading agent that analyzes market narratives and their associated price movements to make trading decisions, with a web-based UI (http://34.216.205.222:8000/) for configuration, backtesting and real-time performance visualization.

## Overview

The Narrative Agent uses historical patterns in market narratives (news, events, sentiment) to predict future price movements. When a new narrative appears that is similar to historical narratives, the agent uses the correlation between past price positions and returns to generate trading signals.

### Key Features

- **Pattern Recognition**: Identifies similar narratives based on keyword overlap
- **Price Position Analysis**: Calculates relative price positions over configurable look-back periods
- **Correlation-Based Signals**: Uses Pearson correlation between price positions and returns
- **Enhanced DeFi Transaction Cost Modeling**: 
  - Realistic AMM pool slippage simulation
  - Dynamic gas fee calculations
  - Protocol fee accounting
- **Position Management**: Automatic position entry/exit with configurable hold periods
- **Risk Management**: Built-in stop loss/gain and comprehensive risk analysis
- **Performance Analytics**: Comprehensive metrics including returns, drawdown, and volatility

## Enhanced Transaction Cost Model

The system includes an enhanced transaction cost model designed for DeFi trading:

- **AMM Pool Simulation**: Models constant product (x*y=k) pools that auto-adjust to external prices
- **Dynamic Slippage Calculation**: Slippage varies based on trade size relative to pool liquidity
- **Gas Fee Tracking**: Configurable gas costs per transaction
- **Cost Breakdown Analysis**: Detailed breakdown of gas, slippage and protocol fees

## Trading Strategy

The Narrative Agent employs a composite strategy that combines multiple signal generators:

### Signal Generators

1. **Pattern Signal**: Correlation-based approach analyzing historical price positions and returns
2. **Sentiment-Event Signal**: Analyzes performance of specific sentiment-event combinations (e.g. bullish macro events)
3. **Price Momentum Signal**: Compares short-term vs long-term price momentum
4. **Sentiment Momentum Signal**: Tracks shifts in overall market narrative sentiment
5. **Volatility Regime Signal**: Adjusts position confidence based on market volatility
6. **Narrative Clustering Signal**: Groups similar narratives and analyzes performance in different market regimes

### Composite Scoring

The agent combines all signals using a weighted scoring system:
- Each signal generates a score between -1 and 1
- Weights are dynamically adjusted based on market conditions:
  - **Bullish markets**: Higher weight on momentum signals
  - **Bearish markets**: Higher weight on sentiment and volatility signals
  - **Ranging markets**: Balanced weights across all signals
- Final position decisions require at least 2 valid signals

### Market Condition Detection

The system continuously monitors market conditions using:
- Multiple moving averages (24h, 72h, 168h)
- Price position relative to MAs
- MA alignment (bullish when shorter > longer)

## SentiChain Analytics and Data

The Narrative Agent directly pulls SentiChain's pre-processed market narratives (this process involves SentiChain's in-house AI Agents) and minute-level price data.

## Web UI

- **Interactive Configuration**: Configure agents through an intuitive web interface
- **Real-time Backtesting**: Watch your agent's performance update live during backtests
- **Performance Visualization**: Interactive charts showing returns, drawdown, and other metrics
- **Transaction Cost Analysis**: Visual breakdown of trading costs

## Installation

### Quick Start with UI (Recommended)

#### Docker
```bash
git clone https://github.com/ruitao-edward-chen/narrative_agent.git
cd narrative_agent
docker-compose up -d
```

Then open your browser to `http://localhost:8000`

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from narrative_agent import NarrativeAgent, NarrativeAgentConfig

# Configure the agent (enhanced transaction cost mode)
config = NarrativeAgentConfig(
    ticker="BTC",
    look_back_period=6,
    hold_period=1,
    count_common_threshold=5,
    use_enhanced_costs=True,
    gas_fee_usd=50.0,
    amm_liquidity_usd=1_000_000.0,
    position_size_usd=10_000.0,
)

# Create agent
agent = NarrativeAgent(config, api_key="<your-sentichain-api-key>")

# Run daily updates
agent.update("2025-02-01T00:00:00")

# Get performance metrics
df = agent.get_performance_dataframe()

# Get transaction cost analysis
cost_summary = agent.get_transaction_cost_summary()
```

## Trading Logic

The enhanced trading logic follows these steps:

1. **Narrative Detection**: When a new narrative with a pattern is detected, the agent activates its composite strategy
2. **Multi-Signal Analysis**: 
   - Pattern correlation analysis
   - Sentiment-event historical performance
   - Price and sentiment momentum calculation
   - Volatility regime assessment
   - Narrative clustering and similarity matching
3. **Market Context**: Determine current market condition (bullish/bearish/ranging)
4. **Composite Score**: Calculate weighted average of all signals with dynamic weights
5. **Signal Generation**:
   - Long (1): Composite score > 0.25
   - Short (-1): Composite score < -0.25
   - No position (0): Otherwise

## Position Management

- Only one position can be active at a time
- New positions (1 or -1) override existing positions
- Positions are automatically closed after the hold period
- Transaction costs are applied based on the enhanced model (gas + slippage + fees)

## Running a Backtest

```bash
python examples/backtest_example.py
```

The enhanced backtest example includes:
- Realistic transaction cost modeling
- Detailed cost breakdown analysis
- Multiple visualization charts
- Performance metrics with cost impact

## Using the Web UI

Once the application is running:

1. **Configure Your Agent**:
   - Select ticker (BTC)
   - Set look-back and hold periods
   - Configure enhanced transaction costs:
     - Gas fee (USD per transaction)
     - AMM pool liquidity
     - Position size
   - Set stop loss and stop gain (optional)
   - Enter your SentiChain API key

2. **Run Backtest**:
   - Choose start date and duration
   - Click "Start Backtest"
   - Watch real-time performance updates

3. **Analyze Results**:
   - View live updating performance charts
   - Monitor key metrics (returns, drawdown, Sharpe ratio)
   - Analyze transaction cost breakdown
   - Compare multiple backtests

## API Requirements

The agent requires a SentiChain API key for accessing narrative and price data. Sign up at [https://sentichain.com](https://sentichain.com) to get your API key.

## Performance Metrics

The agent tracks and reports:
- Total and per-position returns
- Maximum drawdown
- Annualized volatility
- Win rate
- Average return per position
- Transaction cost analysis:
  - Total gas fees paid
  - Average slippage per trade
  - Cost breakdown by type
  - Impact on net returns