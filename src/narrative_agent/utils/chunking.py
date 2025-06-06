"""
Utilities for chunking block ranges.
"""

from typing import List, Tuple


def calculate_chunk_ranges(
    block_number_start: int, block_number_end: int, chunk_size: int = 50
) -> List[Tuple[int, int]]:
    """
    Calculate a list of (chunk_start, chunk_end) tuples covering a block range.

    Args:
        block_number_start: Starting block number (inclusive)
        block_number_end: Ending block number (inclusive)
        chunk_size: Size of each chunk (default: 50)

    Returns:
        List of (chunk_start, chunk_end) tuples

    Raises:
        ValueError: If block_number_end < block_number_start
    """
    if block_number_end < block_number_start:
        raise ValueError("block_number_end must be >= block_number_start")

    start_chunk_index = block_number_start // chunk_size
    end_chunk_index = block_number_end // chunk_size

    chunk_ranges: List[Tuple[int, int]] = []
    for chunk_idx in range(start_chunk_index, end_chunk_index + 1):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size - 1
        chunk_ranges.append((chunk_start, chunk_end))

    return chunk_ranges
