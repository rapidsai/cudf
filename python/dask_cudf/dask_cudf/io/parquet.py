import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

import cudf


class CudfEngine(ArrowEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        meta, stats, parts = ArrowEngine.read_metadata(*args, **kwargs)
        meta = cudf.DataFrame.from_pandas(meta)
        return (meta, stats, parts)

    @staticmethod
    def read_partition(
        fs, piece, columns, index, categories=(), partitions=(), **kwargs
    ):
        if isinstance(index, list):
            columns += index

        df = cudf.read_parquet(
            piece.path,
            engine="cudf",
            columns=columns,
            row_group=piece.row_group,
            **kwargs.get("read", {}),
        )

        if any(index) in df.columns:
            df = df.set_index(index)

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
