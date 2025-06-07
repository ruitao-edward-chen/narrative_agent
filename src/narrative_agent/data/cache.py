"""
Data caching system for narrative and price data.
"""

import hashlib
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pickle


class DataCache:
    """
    Cache manager for narrative and price data.
    """

    def __init__(self, cache_dir: str = ".narrative_cache"):
        """
        Initialize the cache manager.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create subdirectories for different data types
        (self.cache_dir / "narratives").mkdir(exist_ok=True)
        (self.cache_dir / "prices").mkdir(exist_ok=True)

    def _get_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Generate a unique cache key for the given parameters.
        """
        key_string = f"{ticker}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def save_narratives(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        narratives: List[Dict[str, Any]],
    ) -> None:
        """
        Save narratives to cache.
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / "narratives" / f"{cache_key}.pkl"

        cache_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "narratives": narratives,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        print(
            f"Cached {len(narratives)} narratives for {ticker} ({start_date} to {end_date})"
        )

    def load_narratives(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load narratives from cache if available.
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / "narratives" / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache validity
            if (
                cache_data["ticker"] == ticker
                and cache_data["start_date"] == start_date
                and cache_data["end_date"] == end_date
            ):
                print(
                    f"Loaded {len(cache_data['narratives'])} cached narratives for {ticker}"
                )
                return cache_data["narratives"]
        except Exception as e:
            print(f"Error loading narrative cache: {e}")

        return None

    def save_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        prices: List[Tuple[str, float]],
    ) -> None:
        """
        Save prices to cache.
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / "prices" / f"{cache_key}.pkl"

        cache_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "prices": prices,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        print(f"Cached {len(prices)} prices for {ticker} ({start_date} to {end_date})")

    def load_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Load prices from cache if available.
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / "prices" / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache validity
            if (
                cache_data["ticker"] == ticker
                and cache_data["start_date"] == start_date
                and cache_data["end_date"] == end_date
            ):
                print(f"Loaded {len(cache_data['prices'])} cached prices for {ticker}")
                return cache_data["prices"]
        except Exception as e:
            print(f"Error loading price cache: {e}")

        return None

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            (self.cache_dir / "narratives").mkdir(exist_ok=True)
            (self.cache_dir / "prices").mkdir(exist_ok=True)
            print("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        """
        narrative_files = list((self.cache_dir / "narratives").glob("*.pkl"))
        price_files = list((self.cache_dir / "prices").glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in narrative_files + price_files)

        return {
            "cache_dir": str(self.cache_dir),
            "narrative_files": len(narrative_files),
            "price_files": len(price_files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": {
                "narratives": [f.name for f in narrative_files],
                "prices": [f.name for f in price_files],
            },
        }
