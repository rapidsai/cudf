import warnings
from functools import partial

import pyarrow.parquet as pq
from pyarrow.compat import guid

import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

import cudf
from cudf.core.column import build_categorical_column


def _mkdir_if_not_exists(fs, path):
    if fs._isfilestore() and not fs.exists(path):
        try:
            fs.mkdir(path)
        except OSError:
            assert fs.exists(path)


# Mostly borrowed from...
# https://arrow.apache.org/
# docs/_modules/pyarrow/parquet.html#write_to_dataset
def write_to_dataset(
    df, root_path, partition_cols=None, fs=None, preserve_index=False, **kwargs
):
    """Wrapper around parquet.write_table for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
    manner:

    root_dir/
      group1=value1
        group2=value1
          <uuid>.parquet
        group2=value2
          <uuid>.parquet
      group1=valueN
        group2=value1
          <uuid>.parquet
        group2=valueN
          <uuid>.parquet

    Parameters
    ----------
    table : pyarrow.Table
    root_path : string,
        The root directory of the dataset
    filesystem : FileSystem, default None
        If nothing passed, paths assumed to be found in the local on-disk
        filesystem
    partition_cols : list,
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given
    **kwargs : dict,
        kwargs for write_table function. Using `metadata_collector` in
        kwargs allows one to collect the file metadata instances of
        dataset pieces. See docstring for `write_table` or
        `ParquetWriter` for more information.
    """
    _mkdir_if_not_exists(fs, root_path)

    if partition_cols is not None and len(partition_cols) > 0:

        if len(partition_cols) > 1:
            raise ValueError(
                "Only single-column is supported with cudf (for now)"
            )

        data_cols = df.columns.drop(partition_cols)
        if len(data_cols) == 0:
            raise ValueError("No data left to save outside partition columns")

        #  Won't work for multiple `partition_cols`..
        for i, sub_df in enumerate(
            df.scatter_by_map(partition_cols[0], keep_index=preserve_index)
        ):
            if sub_df is None or len(sub_df) < 1:
                continue
            keys = sub_df[partition_cols[0]].iloc[0]
            if not isinstance(keys, tuple):
                keys = (keys,)
            subdir = "/".join(
                [
                    "{colname}={value}".format(colname=name, value=val)
                    for name, val in zip(partition_cols, keys)
                ]
            )
            prefix = "/".join([root_path, subdir])
            _mkdir_if_not_exists(fs, prefix)
            outfile = guid() + ".parquet"
            full_path = "/".join([prefix, outfile])
            sub_df.drop(columns=partition_cols).to_parquet(full_path, **kwargs)
    else:
        outfile = guid() + ".parquet"
        full_path = "/".join([root_path, outfile])
        df.to_parquet(full_path, **kwargs)


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
            # `piece` = (path, row_group, partition_keys)
            (path, row_group, partition_keys) = piece
            piece = pq.ParquetDatasetPiece(
                path,
                row_group=row_group,
                partition_keys=partition_keys,
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

        if index and index[0] in df.columns:
            df = df.set_index(index[0])

        if len(piece.partition_keys) > 0:
            if partitions is None:
                raise ValueError("Must pass partition sets")
            for i, (name, index2) in enumerate(piece.partition_keys):
                categories = [
                    val.as_py() for val in partitions.levels[i].dictionary
                ]
                sr = cudf.Series(index2).astype(type(index2)).repeat(len(df))
                df[name] = build_categorical_column(
                    categories=categories, codes=sr._column, ordered=False
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
        # TODO: Replace `pq.write_table` with gpu-accelerated
        #       write after cudf.io.to_parquet is supported.

        preserve_index = False
        if index_cols:
            df = df.set_index(index_cols)
            preserve_index = True

        use_arrow = return_metadata or (partition_on and len(partition_on) > 1)

        # NOTE: `to_arrow` does not accept `schema` argument
        if use_arrow:
            md_list = []
            t = df.to_arrow(preserve_index=preserve_index)
        else:
            md_list = [None]
        if partition_on:
            if use_arrow:
                pq.write_to_dataset(
                    t,
                    path,
                    partition_cols=partition_on,
                    filesystem=fs,
                    metadata_collector=md_list,
                    **kwargs,
                )
            else:
                write_to_dataset(
                    df,
                    path,
                    partition_cols=partition_on,
                    metadata_collector=md_list,
                    fs=fs,
                    preserve_index=preserve_index,
                    **kwargs,
                )
        else:
            if use_arrow:
                with fs.open(fs.sep.join([path, filename]), "wb") as fil:
                    pq.write_table(
                        t,
                        fil,
                        compression=compression,
                        metadata_collector=md_list,
                        **kwargs,
                    )
                if md_list:
                    md_list[0].set_file_path(filename)
            else:
                df.to_parquet(
                    fs.sep.join([path, filename]),
                    compression=compression,
                    **kwargs,
                )
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"schema": t.schema, "meta": md_list[0]}]
        else:
            return []


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
