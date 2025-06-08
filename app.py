"""
FastAPI backend for Narrative Agent Marketplace
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import hashlib

from narrative_agent import NarrativeAgent, NarrativeAgentConfig


app = FastAPI(title="Narrative Agent Marketplace", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active backtests
active_backtests: Dict[str, Dict[str, Any]] = {}

# Thread pool for running backtests
executor = ThreadPoolExecutor(max_workers=4)

# Check if frontend build exists and mount static files
frontend_build = Path("frontend/build")
if frontend_build.exists():
    app.mount(
        "/static", StaticFiles(directory=str(frontend_build / "static")), name="static"
    )


class AgentConfig(BaseModel):
    """
    Agent configuration model.
    """

    ticker: str = Field(default="BTC", description="Ticker symbol to trade")

    look_back_period: int = Field(default=6, ge=1, description="Hours to look back")

    hold_period: int = Field(default=1, ge=1, description="Hours to hold position")

    transaction_cost: int = Field(
        default=10, ge=0, description="Transaction cost in basis points (legacy)"
    )

    count_common_threshold: int = Field(
        default=5, ge=1, description="Min common keywords"
    )

    stop_loss: Optional[float] = Field(
        default=None, ge=0, description="Stop loss percentage (e.g., 5 = 5%)"
    )

    stop_gain: Optional[float] = Field(
        default=None, ge=0, description="Stop gain percentage (e.g., 10 = 10%)"
    )

    # Enhanced mode.
    gas_fee_usd: float = Field(
        default=50.0, ge=0, description="Gas fee per transaction in USD"
    )

    amm_liquidity_usd: float = Field(
        default=100_000_000.0, gt=0, description="AMM pool liquidity in USD"
    )

    position_size_usd: float = Field(
        default=10_000.0, gt=0, description="Position size in USD"
    )

    use_enhanced_costs: bool = Field(
        default=True, description="Use enhanced transaction cost model"
    )

    api_key: str = Field(description="SentiChain API key")


class BacktestConfig(BaseModel):
    """
    Backtest config model
    """

    start_date: str = Field(description="Start date in ISO format")
    num_days: int = Field(
        default=30, ge=1, le=365, description="Number of days to backtest"
    )
    agent_config: AgentConfig


class BacktestStatus(BaseModel):
    """
    Backtest status model.
    """

    backtest_id: str

    status: str

    progress: float

    current_day: int

    total_days: int

    timestamp: Optional[datetime] = None

    metrics: Optional[Dict[str, float]] = None

    performance_data: Optional[List[Dict[str, Any]]] = None

    config: Optional[Dict[str, Any]] = None

    transaction_cost_summary: Optional[Dict[str, float]] = None

    cache_info: Optional[Dict[str, Any]] = None  # Add this line


async def run_backtest_async(
    backtest_id: str, config: BacktestConfig, websocket: Optional[WebSocket] = None
):
    """
    Run backtest asynchronously with real-time updates
    """
    try:
        # Update status
        active_backtests[backtest_id]["status"] = "running"

        # Create agent configuration
        agent_config = NarrativeAgentConfig(
            ticker=config.agent_config.ticker,
            look_back_period=config.agent_config.look_back_period,
            hold_period=config.agent_config.hold_period,
            transaction_cost=config.agent_config.transaction_cost,
            count_common_threshold=config.agent_config.count_common_threshold,
            stop_loss=config.agent_config.stop_loss,
            stop_gain=config.agent_config.stop_gain,
            gas_fee_usd=config.agent_config.gas_fee_usd,
            amm_liquidity_usd=config.agent_config.amm_liquidity_usd,
            position_size_usd=config.agent_config.position_size_usd,
            use_enhanced_costs=config.agent_config.use_enhanced_costs,
        )

        api_key_hash = hashlib.sha256(config.agent_config.api_key.encode()).hexdigest()[
            :8
        ]
        cache_dir = f".narrative_cache_{api_key_hash}"
        agent = NarrativeAgent(
            agent_config, config.agent_config.api_key, True, cache_dir
        )

        # Run backtest
        start_date = config.start_date
        num_days = config.num_days

        for day in range(num_days):
            if active_backtests[backtest_id]["status"] == "cancelled":
                break

            timestamp = (
                datetime.fromisoformat(start_date) + timedelta(days=day)
            ).isoformat()

            # Update agent
            await asyncio.to_thread(agent.update, timestamp)

            # Get current performance
            df = agent.get_performance_dataframe()

            # Prepare update
            update = {
                "backtest_id": backtest_id,
                "status": "running",
                "progress": (day + 1) / num_days,
                "current_day": day + 1,
                "total_days": num_days,
                "timestamp": timestamp,
                "config": active_backtests[backtest_id]["config"],
            }

            if not df.empty:
                # Calculate metrics
                annualized_return = float(
                    df["position_return"].mean() * (365 * 24 / agent_config.hold_period)
                )
                annualized_vol = float(df["vol_annualized"].iloc[-1])
                metrics = {
                    "total_positions": len(df),
                    "total_return": float(df["cum_return"].iloc[-1]),
                    "max_drawdown": float(df["max_drawdown"].iloc[-1]),
                    "volatility": annualized_vol,
                    "win_rate": float((df["position_return"] > 0).sum() / len(df)),
                    "avg_return": float(df["position_return"].mean()),
                    "annualized_return": annualized_return,
                    "sharpe_ratio": (
                        annualized_return / annualized_vol if annualized_vol > 0 else 0
                    ),
                }

                # Get performance data for chart
                performance_data = []
                for idx, row in df.iterrows():
                    perf_row = {
                        "position": idx,
                        "entry_timestamp": row["entry_timestamp"],
                        "exit_timestamp": row["close_timestamp"],
                        "position_return": float(row["position_return"]),
                        "cum_return": float(row["cum_return"]),
                        "max_drawdown": float(row["max_drawdown"]),
                    }

                    # Add enhanced cost data if available
                    if "total_cost_usd" in row:
                        perf_row.update(
                            {
                                "total_cost_usd": float(row.get("total_cost_usd", 0)),
                                "entry_slippage_bps": float(
                                    row.get("entry_slippage_bps", 0)
                                ),
                                "exit_slippage_bps": float(
                                    row.get("exit_slippage_bps", 0)
                                ),
                            }
                        )

                    performance_data.append(perf_row)

                update["metrics"] = metrics
                update["performance_data"] = performance_data

            # Update status
            active_backtests[backtest_id].update(update)

            # Log progress
            print(
                f"Backtest {backtest_id} progress: {update['progress']:.2%} - Day {day + 1}/{num_days}"
            )

            # Send WebSocket update if connected
            if websocket:
                try:
                    await websocket.send_json(update)
                except Exception as e:
                    print(f"WebSocket send error: {e}")

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)

        # Finalize positions
        final_timestamp = (
            datetime.fromisoformat(start_date) + timedelta(days=num_days)
        ).isoformat()
        await asyncio.to_thread(agent.finalize_positions, final_timestamp)

        # Final update
        df = agent.get_performance_dataframe()
        if not df.empty:
            annualized_return = float(
                df["position_return"].mean() * (365 * 24 / agent_config.hold_period)
            )
            annualized_vol = float(df["vol_annualized"].iloc[-1])
            metrics = {
                "total_positions": len(df),
                "total_return": float(df["cum_return"].iloc[-1]),
                "max_drawdown": float(df["max_drawdown"].iloc[-1]),
                "volatility": annualized_vol,
                "win_rate": float((df["position_return"] > 0).sum() / len(df)),
                "avg_return": float(df["position_return"].mean()),
                "annualized_return": annualized_return,
                "sharpe_ratio": (
                    annualized_return / annualized_vol if annualized_vol > 0 else 0
                ),
            }

            performance_data = []
            for idx, row in df.iterrows():
                perf_row = {
                    "position": idx,
                    "entry_timestamp": row["entry_timestamp"],
                    "exit_timestamp": row["close_timestamp"],
                    "position_return": float(row["position_return"]),
                    "cum_return": float(row["cum_return"]),
                    "max_drawdown": float(row["max_drawdown"]),
                }

                # Add enhanced cost data if available
                if "total_cost_usd" in row:
                    perf_row.update(
                        {
                            "total_cost_usd": float(row.get("total_cost_usd", 0)),
                            "entry_slippage_bps": float(
                                row.get("entry_slippage_bps", 0)
                            ),
                            "exit_slippage_bps": float(row.get("exit_slippage_bps", 0)),
                        }
                    )

                performance_data.append(perf_row)

            active_backtests[backtest_id]["metrics"] = metrics
            active_backtests[backtest_id]["performance_data"] = performance_data

            # Get transaction cost summary if using enhanced model
            if agent_config.use_enhanced_costs:
                cost_summary = agent.get_transaction_cost_summary()
                # Add clarification that this is cumulative across all positions
                if isinstance(cost_summary, dict) and "total_costs" in cost_summary:
                    cost_summary["note"] = (
                        "Cumulative costs across all positions in backtest"
                    )
                active_backtests[backtest_id]["transaction_cost_summary"] = cost_summary

        active_backtests[backtest_id]["status"] = "completed"
        active_backtests[backtest_id]["progress"] = 1.0

        # Send final update
        if websocket:
            try:
                await websocket.send_json(active_backtests[backtest_id])
            except:
                pass

    except Exception as e:
        active_backtests[backtest_id]["status"] = "error"
        active_backtests[backtest_id]["error"] = str(e)
        if websocket:
            try:
                await websocket.send_json(active_backtests[backtest_id])
            except:
                pass


def run_backtest(
    backtest_id: str, agent: NarrativeAgent, start_date: str, num_days: int
):
    """
    Run a backtest synchronously.
    """
    try:
        # Initialize backtest state
        active_backtests[backtest_id]["status"] = "running"
        active_backtests[backtest_id]["current_day"] = 0
        active_backtests[backtest_id]["total_days"] = num_days

        # Get initial cache info
        cache_info = agent.get_cache_info()
        active_backtests[backtest_id]["cache_info"] = cache_info

        # Run the backtest
        for day in range(num_days):
            if active_backtests[backtest_id]["status"] == "cancelled":
                break

            timestamp = (
                datetime.fromisoformat(start_date) + timedelta(days=day)
            ).isoformat()

            # Update agent
            agent.update(timestamp)

            # Update progress
            active_backtests[backtest_id]["progress"] = (day + 1) / num_days
            active_backtests[backtest_id]["current_day"] = day + 1

        # Finalize
        final_timestamp = (
            datetime.fromisoformat(start_date) + timedelta(days=num_days)
        ).isoformat()
        agent.finalize_positions(final_timestamp)

        # Get final results
        df = agent.get_performance_dataframe()
        if not df.empty:
            annualized_return = float(
                df["position_return"].mean() * (365 * 24 / agent.config.hold_period)
            )
            annualized_vol = float(df["vol_annualized"].iloc[-1])
            metrics = {
                "total_positions": len(df),
                "total_return": float(df["cum_return"].iloc[-1]),
                "max_drawdown": float(df["max_drawdown"].iloc[-1]),
                "volatility": annualized_vol,
                "win_rate": float((df["position_return"] > 0).sum() / len(df)),
                "avg_return": float(df["position_return"].mean()),
                "annualized_return": annualized_return,
                "sharpe_ratio": (
                    annualized_return / annualized_vol if annualized_vol > 0 else 0
                ),
            }

            performance_data = []
            for idx, row in df.iterrows():
                perf_row = {
                    "position": idx,
                    "entry_timestamp": row["entry_timestamp"],
                    "exit_timestamp": row["close_timestamp"],
                    "position_return": float(row["position_return"]),
                    "cum_return": float(row["cum_return"]),
                    "max_drawdown": float(row["max_drawdown"]),
                }

                # Add enhanced cost data if available
                if "total_cost_usd" in row:
                    perf_row.update(
                        {
                            "total_cost_usd": float(row.get("total_cost_usd", 0)),
                            "entry_slippage_bps": float(
                                row.get("entry_slippage_bps", 0)
                            ),
                            "exit_slippage_bps": float(row.get("exit_slippage_bps", 0)),
                        }
                    )

                performance_data.append(perf_row)

            active_backtests[backtest_id]["metrics"] = metrics
            active_backtests[backtest_id]["performance_data"] = performance_data

            # Get transaction cost summary if using enhanced model
            if agent.config.use_enhanced_costs:  # Changed from agent_config
                cost_summary = agent.get_transaction_cost_summary()
                # Add clarification that this is cumulative across all positions
                if isinstance(cost_summary, dict) and "total_costs" in cost_summary:
                    cost_summary["note"] = (
                        "Cumulative costs across all positions in backtest"
                    )
                active_backtests[backtest_id]["transaction_cost_summary"] = cost_summary

        # Update final cache info
        final_cache_info = agent.get_cache_info()
        active_backtests[backtest_id]["cache_info"] = final_cache_info

        active_backtests[backtest_id]["status"] = "completed"
        active_backtests[backtest_id]["progress"] = 1.0

    except Exception as e:
        active_backtests[backtest_id]["status"] = "error"
        active_backtests[backtest_id]["error"] = str(e)


@app.get("/api")
async def api_info():
    """
    API info endpoint
    """
    return {"message": "Narrative Agent Marketplace API", "version": "1.0.0"}


# Serve the frontend for root and non-API routes
@app.get("/")
async def serve_root():
    """
    Serve the frontend app
    """
    if frontend_build.exists():
        return FileResponse(str(frontend_build / "index.html"))
    return {
        "message": "Frontend not built. Please run: cd frontend && npm install && npm run build"
    }


@app.post("/backtest/start")
async def start_backtest(
    request: BacktestConfig,
):
    """
    Start a new backtest.
    """
    try:
        # Create unique backtest ID
        backtest_id = str(uuid.uuid4())

        # Create agent configuration
        agent_config = NarrativeAgentConfig(
            ticker=request.agent_config.ticker,
            look_back_period=request.agent_config.look_back_period,
            hold_period=request.agent_config.hold_period,
            transaction_cost=request.agent_config.transaction_cost,
            count_common_threshold=request.agent_config.count_common_threshold,
            stop_loss=request.agent_config.stop_loss,
            stop_gain=request.agent_config.stop_gain,
            use_enhanced_costs=request.agent_config.use_enhanced_costs,
            gas_fee_usd=request.agent_config.gas_fee_usd,
            amm_liquidity_usd=request.agent_config.amm_liquidity_usd,
            position_size_usd=request.agent_config.position_size_usd,
        )

        # Create API key-based cache directory
        api_key_hash = hashlib.sha256(
            request.agent_config.api_key.encode()
        ).hexdigest()[:8]
        cache_dir = f".narrative_cache_{api_key_hash}"

        # Create agent with API key-specific cache
        agent = NarrativeAgent(
            agent_config,
            request.agent_config.api_key,
            use_cache=True,  # Enable caching
            cache_dir=cache_dir,  # Use API key-specific cache
        )

        # Start backtest
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            executor,
            run_backtest,
            backtest_id,
            agent,
            request.start_date,
            request.num_days,
        )

        # Initialize backtest state
        active_backtests[backtest_id] = {
            "backtest_id": backtest_id,
            "status": "initializing",
            "progress": 0.0,
            "current_day": 0,
            "total_days": request.num_days,
            "config": request.model_dump(),
            "metrics": None,
            "performance_data": None,
            "error": None,
            "cache_info": agent.get_cache_info(),
        }

        return BacktestStatus(**active_backtests[backtest_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{backtest_id}/status", response_model=BacktestStatus)
async def get_backtest_status(backtest_id: str):
    """
    Get backtest status
    """
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return BacktestStatus(**active_backtests[backtest_id])


@app.post("/backtest/{backtest_id}/cancel")
async def cancel_backtest(backtest_id: str):
    """
    Cancel a running backtest
    """
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    active_backtests[backtest_id]["status"] = "cancelled"
    return {"message": "Backtest cancelled"}


@app.get("/backtests", response_model=List[BacktestStatus])
async def list_backtests():
    """
    List all backtests
    """
    return [BacktestStatus(**bt) for bt in active_backtests.values()]


@app.websocket("/ws/{backtest_id}")
async def websocket_endpoint(websocket: WebSocket, backtest_id: str):
    """
    WebSocket endpoint for real-time backtest updates
    """
    print(f"WebSocket connection request for backtest {backtest_id}")
    await websocket.accept()
    print(f"WebSocket connected for backtest {backtest_id}")

    try:
        # Send initial status
        if backtest_id in active_backtests:
            await websocket.send_json(active_backtests[backtest_id])
            print(f"Sent initial status for backtest {backtest_id}")

        # Poll for updates
        last_progress = -1
        while True:
            if backtest_id in active_backtests:
                backtest = active_backtests[backtest_id]
                status = backtest["status"]

                # Send update if progress changed
                if backtest["progress"] != last_progress:
                    last_progress = backtest["progress"]
                    await websocket.send_json(backtest)
                    print(
                        f"WebSocket update sent for {backtest_id}: {last_progress:.2%}"
                    )

                # Exit if backtest is done
                if status in ["completed", "error", "cancelled"]:
                    print(f"Backtest {backtest_id} finished with status: {status}")
                    break
            else:
                print(f"Backtest {backtest_id} not found in active backtests")
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for backtest {backtest_id}")
    except Exception as e:
        print(f"WebSocket error for {backtest_id}: {e}")


# Catch-all route for SPA - must be after all API routes
@app.get("/{path:path}")
async def serve_spa(path: str):
    """
    Serve frontend app for all non-API routes
    """
    if frontend_build.exists():
        file_path = frontend_build / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_build / "index.html"))
    return {"error": "Frontend not found", "path": path}


@app.get("/cache/info/{api_key_prefix}")
async def get_cache_info(api_key_prefix: str):
    """
    Get cache information for an API key.
    """
    try:
        api_key_hash = hashlib.sha256(api_key_prefix.encode()).hexdigest()[:8]
        cache_dir = f".narrative_cache_{api_key_hash}"

        from src.narrative_agent.data.cache import DataCache

        cache = DataCache(cache_dir)

        return cache.get_cache_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear/{api_key_prefix}")
async def clear_cache(api_key_prefix: str):
    """
    Clear cache for an API key.
    """
    try:
        api_key_hash = hashlib.sha256(api_key_prefix.encode()).hexdigest()[:8]
        cache_dir = f".narrative_cache_{api_key_hash}"

        from src.narrative_agent.data.cache import DataCache

        cache = DataCache(cache_dir)
        cache.clear_cache()

        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
