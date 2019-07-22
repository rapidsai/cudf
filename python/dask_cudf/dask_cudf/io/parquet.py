import dask.dataframe as dd
import cudf

try:
    from dask.dataframe.io.parquet.arrow import ArrowEngine

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


except ImportError:
    from glob import glob

    from dask.base import tokenize
    from dask.compatibility import apply
    from dask.utils import natural_sort_key

    ArrowEngine = None


def read_parquet(path, **kwargs):
    """ Read parquet files into a Dask DataFrame

    This calls ``dask.dataframe.read_parquet`` to cordinate the execution of
    ``cudf.read_parquet`` on many distinct parquet files.  The Dask version
    must inherit from the the ``ArrowEngine`` class to support full
    functionality.
    See ``cudf.read_parquet`` and the Dask source code for further details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_parquet("/path/to/dataset/")  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """

    if ArrowEngine:
        columns = kwargs.pop("columns", None)
        if isinstance(columns, str):
            columns = [columns]
        return dd.read_parquet(
            path, columns=columns, engine=CudfEngine, **kwargs
        )

    else:
        name = "read-parquet-" + tokenize(path, **kwargs)

        paths = path
        if isinstance(path, str):
            paths = sorted(glob(str(path)))

        # Ignore *_metadata files for now
        paths = sorted(
            [f for f in paths if not f.endswith("_metadata")],
            key=natural_sort_key,
        )

        # Use 0th file to create meta
        meta = cudf.read_parquet(paths[0], **kwargs)
        graph = {
            (name, i): (apply, cudf.read_parquet, [fn], kwargs)
            for i, fn in enumerate(paths)
        }
        divisions = [None] * (len(paths) + 1)

        return dd.core.new_dd_object(graph, name, meta, divisions)
