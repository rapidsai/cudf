# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# We need to deterministically hash the nodes of the IR graph.
#
# This is used by our multi-partition executor to determine if two IR nodes are
# the same. We need the hashed value to be deterministic (recomputing it on the
# same inputs should yield the same result) and unique (two different inputs
# should yield different hashes).
#
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars.polars


def hash_polars_dataframe(df: polars.polars.PyDataFrame) -> int:
    """
    Compute a deterministic hash for a polars DataFrame.

    Parameters
    ----------
    df : polars.polars.PyDataFrame
        The polars PyDataFrame to hash.

    Returns
    -------
    int
        A deterministic hash value for the DataFrame
    """
    hasher = hashlib.md5(usedforsecurity=False)
    # Convert polars DataFrame to Arrow table (zero-copy operation)
    # breakpoint()
    record_batches = df.to_arrow(compat_level=0)
    for record_batch in record_batches:
        # Initialize hasher with optional seed
        # Add schema hash
        hasher.update(record_batch.schema.serialize())

        # Hash each column's data
        for col in record_batch.columns:
            # This record batch might be a slice of a larger Table. The col
            # contains an integer offset that can be used to disambiguate
            hasher.update(bytes(col.offset))

            # Get buffers from the chunk
            for buffer in col.buffers():
                # Some buffers might be None (e.g., for null buffers when there are no nulls)
                if buffer is not None:
                    # Get the buffer as bytes and update hash
                    # buffer_bytes = buffer.to_pybytes()
                    hasher.update(buffer)

        # Convert the hash digest to an integer
    hash_bytes = hasher.digest()
    return int.from_bytes(hash_bytes, byteorder="little")
