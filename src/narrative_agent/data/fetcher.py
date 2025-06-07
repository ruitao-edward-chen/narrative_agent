"""
Pre-processed data fetching functions for narratives and prices.
"""

from typing import Dict, List, Tuple, Any

from .client import SentiChainClient
from .cache import DataCache
from ..utils import calculate_chunk_ranges


class DataFetcher:
    """
    Pre-processed data fetcher for narratives and prices with caching support.
    """

    def __init__(
        self, api_key: str, use_cache: bool = True, cache_dir: str = ".narrative_cache"
    ):
        """
        Initialize the data fetcher.
        """
        self.client = SentiChainClient(api_key)
        self.use_cache = use_cache
        self.cache = DataCache(cache_dir) if use_cache else None

    def get_narratives(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get narratives for a ticker between start and end dates.
        First checks cache, then fetches from API if needed.
        """
        # Try to load from cache first
        if self.use_cache:
            cached_narratives = self.cache.load_narratives(ticker, start_date, end_date)
            if cached_narratives is not None:
                return cached_narratives

        # Not in cache, fetch from API
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

        # Save to cache
        if self.use_cache and narratives_sorted:
            self.cache.save_narratives(ticker, start_date, end_date, narratives_sorted)

        return narratives_sorted

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Tuple[str, float]]:
        """
        Get prices for a ticker between start and end dates.
        First checks cache, then fetches from API if needed.
        """
        # Try to load from cache first
        if self.use_cache:
            cached_prices = self.cache.load_prices(ticker, start_date, end_date)
            if cached_prices is not None:
                return cached_prices

        # Not in cache, fetch from API
        prices_data = self.client.fetch_prices(ticker, start_date, end_date)

        if not prices_data:
            print(
                f"Failed to fetch prices for {ticker} from {start_date} to {end_date}"
            )
            return []

        # Extract close prices
        prices = []
        if isinstance(prices_data, dict):
            for timestamp, price_dict in prices_data.items():
                if isinstance(price_dict, dict) and "c" in price_dict:
                    prices.append((timestamp, price_dict["c"]))
        else:
            print(f"Unexpected price data format: {type(prices_data)}")
            return []

        # Sort by timestamp
        prices_sorted = sorted(prices, key=lambda x: x[0])

        # Save to cache
        if self.use_cache and prices_sorted:
            self.cache.save_prices(ticker, start_date, end_date, prices_sorted)

        return prices_sorted

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        if self.cache:
            self.cache.clear_cache()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        """
        if self.cache:
            return self.cache.get_cache_info()
        return {"message": "Caching is disabled"}
