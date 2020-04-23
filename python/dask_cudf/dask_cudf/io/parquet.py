import warnings
from functools import partial

import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

import cudf
from cudf.core.column import as_column, build_categorical_column
from cudf.io import write_to_dataset


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
            path = piece
            row_group = None
            partition_keys = []
        else:
            (path, row_group, partition_keys) = piece

        strings_to_cats = kwargs.get("strings_to_categorical", False)
        if cudf.utils.ioutils._is_local_filesystem(fs):
            df = cudf.read_parquet(
                path,
                engine="cudf",
                columns=columns,
                row_group=row_group,
                strings_to_categorical=strings_to_cats,
                **kwargs.get("read", {}),
            )
        else:
            with fs.open(path, mode="rb") as f:
                df = cudf.read_parquet(
                    f,
                    engine="cudf",
                    columns=columns,
                    row_group=row_group,
                    strings_to_categorical=strings_to_cats,
                    **kwargs.get("read", {}),
                )

        if index and index[0] in df.columns:
            df = df.set_index(index[0])

        if len(partition_keys) > 0:
            if partitions is None:
                raise ValueError("Must pass partition sets")
            for i, (name, index2) in enumerate(partition_keys):
                categories = [
                    val.as_py() for val in partitions.levels[i].dictionary
                ]
                sr = cudf.Series(index2).astype(type(index2)).repeat(len(df))
                df[name] = build_categorical_column(
                    categories=categories,
                    codes=as_column(
                        sr._column.base_data, dtype=sr._column.dtype
                    ),
                    size=sr._column.size,
                    offset=sr._column.offset,
                    ordered=False,
                )

        return df

    @staticmethod
    def write_partition(
        df,
        path,
        fs,
        filename,
        partition_on,
        return_metadata,
        fmd=None,
        compression=None,
        index_cols=None,
        **kwargs,
    ):
        preserve_index = False
        if partition_on:
            md = write_to_dataset(
                df,
                path,
                partition_cols=partition_on,
                fs=fs,
                preserve_index=preserve_index,
                return_metadata=return_metadata,
                **kwargs,
            )
        else:
            md = df.to_parquet(
                fs.sep.join([path, filename]),
                compression=compression,
                metadata_file_path=filename if return_metadata else None,
                **kwargs,
            )
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"meta": md}]
        else:
            return []

    @staticmethod
    def write_metadata(parts, fmd, fs, path, append=False, **kwargs):
        if parts:
            # Aggregate metadata and write to _metadata file
            metadata_path = fs.sep.join([path, "_metadata"])
            _meta = []
            if append and fmd is not None:
                _meta = [fmd]
            _meta.extend([parts[i][0]["meta"] for i in range(len(parts))])
            _meta = (
                cudf.io.merge_parquet_filemetadata(_meta)
                if len(_meta) > 1
                else _meta[0]
            )
            with fs.open(metadata_path, "wb") as fil:
                _meta.tofile(fil)


def read_parquet(
    path,
    columns=None,
    chunksize=None,
    split_row_groups=True,
    gather_statistics=None,
    **kwargs,
):
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
    if isinstance(columns, str):
        columns = [columns]
    if chunksize and gather_statistics is False:
        warnings.warn(
            "Setting chunksize parameter with gather_statistics=False. "
            "Use gather_statistics=True to enable row-group aggregation."
        )
    if chunksize and split_row_groups is False:
        warnings.warn(
            "Setting chunksize parameter with split_row_groups=False. "
            "Use split_row_groups=True to enable row-group aggregation."
        )
    return dd.read_parquet(
        path,
        columns=columns,
        chunksize=chunksize,
        split_row_groups=split_row_groups,
        gather_statistics=gather_statistics,
        engine=CudfEngine,
        **kwargs,
    )


to_parquet = partial(dd.to_parquet, engine=CudfEngine)
