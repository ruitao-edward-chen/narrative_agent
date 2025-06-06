"""
SentiChain API client for fetching narratives and market data.
"""

from typing import Dict, List, Optional, Any
import requests

from ..utils import parse_json_string


class SentiChainClient:
    """
    Client for interacting with the SentiChain API.
    """

    BASE_URL = "https://api.sentichain.com"
    DEFAULT_TIMEOUT = 30

    def __init__(self, api_key: str):
        """
        Initialize the SentiChain client.
        """
        self.api_key = api_key

    def fetch_chain_length(self) -> Optional[int]:
        """
        Fetch the current SentiChain chain length.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/blockchain/get_chain_length?network=mainnet",
                timeout=self.DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("chain_length")
        except Exception as e:
            print(f"Error fetching chain length: {e}")
        return None

    def fetch_narratives(
        self, ticker: str, chunk_start: int, chunk_end: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch narratives for a specific ticker and chunk range.
        """
        summary_type = "l3_event_sentiment_reasoning"
        try:
            resp = requests.get(
                f"{self.BASE_URL}/agent/get_reasoning",
                params={
                    "ticker": ticker,
                    "summary_type": summary_type,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "api_key": self.api_key,
                },
                timeout=self.DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                resp_json_string = resp.json().get("reasoning")
                return parse_json_string(resp_json_string)
        except Exception as e:
            print(f"Error fetching narratives: {e}")
        return None

    def fetch_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Fetch price data for a specific ticker and date range.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/market/get_data",
                params={
                    "ticker": ticker,
                    "sdate": start_date,
                    "edate": end_date,
                    "freq": "minute",
                    "api_key": self.api_key,
                },
                timeout=self.DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json().get("data")
                return data
        except Exception as e:
            print(f"Error fetching prices: {e}")
        return None

    def fetch_timestamp_from_block_number(self, block_number: int) -> Optional[str]:
        """
        Fetch timestamp for a given block number.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/blockchain/get_timestamp_from_block_number",
                params={"network": "mainnet", "block_number": block_number},
                timeout=self.DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("timestamp")
        except Exception as e:
            print(f"Error fetching timestamp from block number: {e}")
        return None

    def fetch_block_number_from_timestamp(self, timestamp: str) -> Optional[int]:
        """
        Fetch block number for a given timestamp.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/blockchain/get_block_number_from_timestamp",
                params={"network": "mainnet", "timestamp": timestamp},
                timeout=self.DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("block_number")
        except Exception as e:
            print(f"Error fetching block number from timestamp: {e}")
        return None
