"""
Pre-processed data fetching functions for narratives and prices.
"""

from typing import Dict, List, Tuple, Any

from .client import SentiChainClient
from ..utils import calculate_chunk_ranges


class DataFetcher:
    """
    Pre-processed data fetcher for narratives and prices.
    """

    def __init__(self, api_key: str):
        """
        Initialize the data fetcher.
        """
        self.client = SentiChainClient(api_key)

    def get_narratives(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get narratives for a ticker between start and end dates.
        """
        # Convert dates to block numbers
        block_number_start = self.client.fetch_block_number_from_timestamp(start_date)
        block_number_end = self.client.fetch_block_number_from_timestamp(end_date)

        if block_number_start is None or block_number_end is None:
            print(
                f"Failed to fetch block numbers for date range: {start_date} to {end_date}"
            )
            return []

        # Calculate chunk ranges
        chunk_ranges = calculate_chunk_ranges(block_number_start, block_number_end)

        # Fetch narratives for each chunk
        narratives: List[Dict[str, Any]] = []
        for chunk_start, chunk_end in chunk_ranges:
            chunk_narratives = self.client.fetch_narratives(
                ticker, chunk_start, chunk_end
            )
            if isinstance(chunk_narratives, list):
                narratives.extend(chunk_narratives)

        # Sort narratives by timestamp
        narratives_sorted = sorted(narratives, key=lambda x: x.get("timestamp", ""))

        return narratives_sorted

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Tuple[str, float]]:
        """
        Get prices for a ticker between start and end dates.
        """
        prices_data = self.client.fetch_prices(ticker, start_date, end_date)

        if not prices_data:
            print(
                f"Failed to fetch prices for {ticker} from {start_date} to {end_date}"
            )
            return []

        # Extract close prices
        prices = []
        for timestamp, price_dict in prices_data.items():
            if isinstance(price_dict, dict) and "c" in price_dict:
                prices.append((timestamp, price_dict["c"]))

        # Sort by timestamp
        prices_sorted = sorted(prices, key=lambda x: x[0])

        return prices_sorted
