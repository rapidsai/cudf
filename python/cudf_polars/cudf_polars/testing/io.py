# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""IO testing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    import polars as pl

__all__: list[str] = ["make_partitioned_source"]


def make_partitioned_source(
    df: pl.DataFrame,
    path: str | Path,
    fmt: Literal["csv", "ndjson", "parquet", "chunked_parquet"],
    *,
    n_files: int = 1,
    row_group_size: int | None = None,
    write_kwargs: dict | None = None,
) -> None:
    """
    Write the Polars DataFrame to one or more files of the desired format.

    Parameters
    ----------
    df : polars.DataFrame
        The input DataFrame to write.
    path : str | pathlib.Path
        The base path to write the file(s) to.
    fmt : Literal["csv", "ndjson", "parquet", "chunked_parquet"]
        The format to write in.
    n_files : int, default 1
        If greater than 1, splits the data into multiple files.
    row_group_size : optional, int
        Only used for Parquet. Specifies the row group size per file.
    write_kwargs : dict, optional
        Additional keyword arguments to pass to the write_* functions.
    """
    path = Path(path)
    write_kwargs = write_kwargs or {}

    def write(part: pl.DataFrame, file_path: Path) -> None:
        match fmt:
            case "csv":
                part.write_csv(file_path, **write_kwargs)
            case "ndjson":
                part.write_ndjson(file_path, **write_kwargs)
            case "parquet" | "chunked_parquet":
                part.write_parquet(
                    file_path,
                    row_group_size=row_group_size or (len(part) // 2),
                    **write_kwargs,
                )
            case _:
                raise ValueError(f"Unsupported format: {fmt}")

    if n_files == 1:
        if path.is_dir():
            path = path / f"part.0.{fmt}"
        write(df, path)
    else:
        stride = len(df) // n_files
        for i, part in enumerate(df.iter_slices(stride)):
            file_path = path / f"part.{i}.{fmt}"
            write(part, file_path)
