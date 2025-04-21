# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""IO testing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def make_partitioned_source(
    df: pl.DataFrame,
    path: str | Path,
    fmt: str,
    *,
    n_files: int = 1,
    row_group_size: int | None = None,
) -> None:
    """
    Write the Polars DataFrame to one or more files of the desired format.

    Parameters
    ----------
    df : polars.DataFrame
        The input DataFrame to write.
    path : str | pathlib.Path
        The base path to write the file(s) to.
    fmt : str
        The format to write in.
    n_files : int, default 1
        If greater than 1, splits the data into multiple files.
    row_group_size : optional, int
        Only used for Parquet. Specifies the row group size per file.
    """
    path = Path(path)
    if n_files == 1:
        if fmt == "csv":
            df.write_csv(path)
        elif fmt == "ndjson":
            df.write_ndjson(path)
        elif fmt in {"parquet", "chunked_parquet"}:
            df.write_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    else:
        n_rows = len(df)
        stride = int(n_rows / n_files)
        for i in range(n_files):
            offset = stride * i
            part = df.slice(offset, stride)
            file_path = path / f"part.{i}.{fmt}"
            if fmt == "csv":
                part.write_csv(file_path)
            elif fmt == "ndjson":
                part.write_ndjson(file_path)
            elif fmt == "parquet":
                part.write_parquet(
                    file_path,
                    row_group_size=row_group_size or int(stride / 2),
                )
            else:
                raise ValueError(f"Unsupported format: {fmt}")
