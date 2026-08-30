"""Random access to TORGO audio bytes inside the cached parquet shards.

`sample_id` is `lt-<global row index>` over the shards sorted by filename, which
is exactly how scripts/lt_metadata.py assigned them. Rows are fetched a row
group at a time so a handful of samples never forces a whole 370 MB shard
through memory.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent
HF_CACHE = PROJECT / ".hf_cache" / "hub" / "datasets--abnerh--TORGO-database"


@lru_cache(maxsize=1)
def _shards() -> list[tuple[Path, int, list[int]]]:
    """(path, first global row index, cumulative row-group offsets)."""
    out, start = [], 0
    for path in sorted(HF_CACHE.rglob("*.parquet")):
        meta = pq.ParquetFile(path).metadata
        offsets, running = [], 0
        for i in range(meta.num_row_groups):
            offsets.append(running)
            running += meta.row_group(i).num_rows
        out.append((path, start, offsets))
        start += meta.num_rows
    return out


def _locate(global_index: int) -> tuple[Path, int, int]:
    """-> (shard path, row-group number, row offset inside that row group)."""
    for path, start, offsets in _shards():
        total = offsets[-1] + pq.ParquetFile(path).metadata.row_group(
            len(offsets) - 1).num_rows
        if global_index < start + total:
            local = global_index - start
            for group in range(len(offsets) - 1, -1, -1):
                if local >= offsets[group]:
                    return path, group, local - offsets[group]
    raise IndexError(f"row {global_index} is past the end of the dataset")


def audio_bytes_for(sample_ids: list[str]) -> dict[str, bytes]:
    """Fetch WAV bytes for the given sample_ids, batched by row group."""
    wanted: dict[tuple[Path, int], list[tuple[int, str]]] = defaultdict(list)
    for sample_id in sample_ids:
        path, group, offset = _locate(int(sample_id.split("-")[1]))
        wanted[(path, group)].append((offset, sample_id))

    out: dict[str, bytes] = {}
    for (path, group), rows in wanted.items():
        table = pq.ParquetFile(path).read_row_group(group, columns=["audio"])
        column = table.column("audio").to_pylist()
        for offset, sample_id in rows:
            out[sample_id] = column[offset]["bytes"]
    return out


def iter_audio(sample_ids: list[str], batch_size: int = 64):
    """Yield (sample_id, wav_bytes) in the order given, a batch at a time."""
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i + batch_size]
        loaded = audio_bytes_for(batch)
        for sample_id in batch:
            yield sample_id, loaded[sample_id]
