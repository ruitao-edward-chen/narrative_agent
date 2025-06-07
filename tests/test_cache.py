"""
Test data caching functionality.
"""

import os
import sys
import time
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.narrative_agent.data import DataFetcher


def test_caching():
    """
    Test data caching functionality.
    """
    api_key = os.getenv("SENTICHAIN_API_KEY")
    if not api_key:
        print("ERROR: SENTICHAIN_API_KEY environment variable not set!")
        return

    # Test parameters
    ticker = "BTC"
    start_date = "2025-01-01T00:00:00"
    end_date = "2025-01-02T00:00:00"

    print("=== Data Caching Test ===\n")

    # Create fetcher with caching enabled
    fetcher = DataFetcher(api_key, use_cache=True)

    # Show initial cache info
    cache_info = fetcher.get_cache_info()
    print("Initial cache state:")
    print(f"  - Directory: {cache_info['cache_dir']}")
    print(f"  - Narrative files: {cache_info['narrative_files']}")
    print(f"  - Price files: {cache_info['price_files']}")
    print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")

    # First fetch
    print("\n1. First fetch (API call)...")
    start_time = time.time()
    narratives = fetcher.get_narratives(ticker, start_date, end_date)
    prices = fetcher.get_prices(ticker, start_date, end_date)
    first_fetch_time = time.time() - start_time

    print(f"   - Fetched {len(narratives)} narratives")
    print(f"   - Fetched {len(prices)} prices")
    print(f"   - Time taken: {first_fetch_time:.2f} seconds")

    # Show cache info after first fetch
    cache_info = fetcher.get_cache_info()
    print("\nCache after first fetch:")
    print(f"  - Narrative files: {cache_info['narrative_files']}")
    print(f"  - Price files: {cache_info['price_files']}")
    print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")

    # Second fetch
    print("\n2. Second fetch (from cache)...")
    start_time = time.time()
    narratives2 = fetcher.get_narratives(ticker, start_date, end_date)
    prices2 = fetcher.get_prices(ticker, start_date, end_date)
    second_fetch_time = time.time() - start_time

    print(f"   - Loaded {len(narratives2)} narratives")
    print(f"   - Loaded {len(prices2)} prices")
    print(f"   - Time taken: {second_fetch_time:.2f} seconds")
    print(f"   - Speedup: {first_fetch_time/second_fetch_time:.1f}x faster")

    # Verify data consistency
    print("\n3. Data consistency check...")
    narratives_match = len(narratives) == len(narratives2)
    prices_match = len(prices) == len(prices2)
    print(f"   - Narratives match: {narratives_match}")
    print(f"   - Prices match: {prices_match}")

    # Test with different date range
    print("\n4. Different date range (new API call)...")
    different_end = "2025-01-03T00:00:00"
    start_time = time.time()
    narratives3 = fetcher.get_narratives(ticker, start_date, different_end)
    third_fetch_time = time.time() - start_time

    print(f"   - Fetched {len(narratives3)} narratives")
    print(f"   - Time taken: {third_fetch_time:.2f} seconds")

    # Final cache info
    cache_info = fetcher.get_cache_info()
    print("\nFinal cache state:")
    print(f"  - Narrative files: {cache_info['narrative_files']}")
    print(f"  - Price files: {cache_info['price_files']}")
    print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")
    print(f"  - Files: {cache_info['files']}")

    # Demonstrate cache control
    print("\n5. Cache control options...")

    # Create fetcher without caching
    _ = DataFetcher(api_key, use_cache=False)
    print("   - Created fetcher with caching disabled")

    # Clear cache
    if input("\nClear cache? (y/n): ").lower() == "y":
        fetcher.clear_cache()
        print("   - Cache cleared!")

        cache_info = fetcher.get_cache_info()
        print("\nCache after clearing:")
        print(f"  - Narrative files: {cache_info['narrative_files']}")
        print(f"  - Price files: {cache_info['price_files']}")
        print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    test_caching()
