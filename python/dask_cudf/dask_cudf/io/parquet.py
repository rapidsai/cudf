from functools import partial

import pyarrow.parquet as pq

import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

import cudf
from cudf.core.column import CategoricalColumn


class CudfEngine(ArrowEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        meta, stats, parts = ArrowEngine.read_metadata(*args, **kwargs)

        # If `strings_to_categorical==True`, convert objects to int32
        strings_to_cats = kwargs.get("strings_to_categorical", False)
        dtypes = {}
        for col in meta.columns:
            if meta[col].dtype == "O":
                dtypes[col] = "int32" if strings_to_cats else "object"

        meta = cudf.DataFrame.from_pandas(meta)
        for col, dtype in dtypes.items():
            meta[col] = meta[col].astype(dtype)

        return (meta, stats, parts)

    @staticmethod
    def read_partition(
        fs, piece, columns, index, categories=(), partitions=(), **kwargs
    ):
        if columns is not None:
            columns = [c for c in columns]
        if isinstance(index, list):
            columns += index

        if isinstance(piece, str):
            # `piece` is a file-path string
            piece = pq.ParquetDatasetPiece(
                piece, open_file_func=partial(fs.open, mode="rb")
            )
        else:
            # `piece` contains (path, row_group, partition_keys)
            piece = pq.ParquetDatasetPiece(
                piece[0],
                row_group=piece[1],
                partition_keys=piece[2],
                open_file_func=partial(fs.open, mode="rb"),
            )

        strings_to_cats = kwargs.get("strings_to_categorical", False)
        if cudf.utils.ioutils._is_local_filesystem(fs):
            df = cudf.read_parquet(
                piece.path,
                engine="cudf",
                columns=columns,
                row_group=piece.row_group,
                strings_to_categorical=strings_to_cats,
                **kwargs.get("read", {}),
            )
        else:
            with fs.open(piece.path, mode="rb") as f:
                df = cudf.read_parquet(
                    f,
                    engine="cudf",
                    columns=columns,
                    row_group=piece.row_group,
                    strings_to_categorical=strings_to_cats,
                    **kwargs.get("read", {}),
                )

        if index is not None and index[0] in df.columns:
            df = df.set_index(index[0])

        if len(piece.partition_keys) > 0:
            if partitions is None:
                raise ValueError("Must pass partition sets")
            for i, (name, index2) in enumerate(piece.partition_keys):
                categories = [
                    val.as_py() for val in partitions.levels[i].dictionary
                ]
                sr = cudf.Series(index2).astype(type(index2)).repeat(len(df))
                df[name] = CategoricalColumn(
                    data=sr._column.data, categories=categories, ordered=False
                )

        return df


def read_parquet(path, **kwargs):
    """ Read parquet files into a Dask DataFrame

    Calls ``dask.dataframe.read_parquet`` to cordinate the execution of
    ``cudf.read_parquet``, and ultimately read multiple partitions into a
    single Dask dataframe. The Dask version must supply an ``ArrowEngine``
    class to support full functionality.
    See ``cudf.read_parquet`` and Dask documentation for further details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_parquet("/path/to/dataset/")  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """

    columns = kwargs.pop("columns", None)
    if isinstance(columns, str):
        columns = [columns]
    return dd.read_parquet(path, columns=columns, engine=CudfEngine, **kwargs)
