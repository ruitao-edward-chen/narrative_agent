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
    # Mount static files
    app.mount(
        "/static", StaticFiles(directory=str(frontend_build / "static")), name="static"
    )


class AgentConfig(BaseModel):
    """Agent configuration model"""

    ticker: str = Field(default="BTC", description="Ticker symbol to trade")
    look_back_period: int = Field(default=6, ge=1, description="Hours to look back")
    hold_period: int = Field(default=1, ge=1, description="Hours to hold position")
    transaction_cost: int = Field(
        default=10, ge=0, description="Transaction cost in basis points"
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
    Backtest status model
    """

    backtest_id: str
    status: str
    progress: float
    current_day: int
    total_days: int
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    performance_data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


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
        )

        agent = NarrativeAgent(agent_config, config.agent_config.api_key)

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
                    performance_data.append(
                        {
                            "position": idx,
                            "entry_timestamp": row["entry_timestamp"],
                            "exit_timestamp": row["close_timestamp"],
                            "position_return": float(row["position_return"]),
                            "cum_return": float(row["cum_return"]),
                            "max_drawdown": float(row["max_drawdown"]),
                        }
                    )

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
                performance_data.append(
                    {
                        "position": idx,
                        "entry_timestamp": row["entry_timestamp"],
                        "exit_timestamp": row["close_timestamp"],
                        "position_return": float(row["position_return"]),
                        "cum_return": float(row["cum_return"]),
                        "max_drawdown": float(row["max_drawdown"]),
                    }
                )

            active_backtests[backtest_id]["metrics"] = metrics
            active_backtests[backtest_id]["performance_data"] = performance_data

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


@app.post("/backtest/start", response_model=BacktestStatus)
async def start_backtest(config: BacktestConfig):
    """
    Start a new backtest
    """
    backtest_id = str(uuid.uuid4())

    # Initialize backtest status
    active_backtests[backtest_id] = {
        "backtest_id": backtest_id,
        "status": "initializing",
        "progress": 0.0,
        "current_day": 0,
        "total_days": config.num_days,
        "config": config.model_dump(),
        "metrics": None,
        "performance_data": None,
        "error": None,
    }

    # Start backtest in background
    asyncio.create_task(run_backtest_async(backtest_id, config))

    return BacktestStatus(**active_backtests[backtest_id])


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

            await asyncio.sleep(0.5)  # Poll every 500ms

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
