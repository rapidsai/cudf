# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

import pyarrow.parquet as pq
from pyarrow.compat import guid

import cudf
import cudf._libxx.parquet as libparquet
from cudf.utils import ioutils


def _get_partition_groups(df, partition_cols, preserve_index=False):
    # TODO: We can use groupby functionality here after cudf#4346.
    #       Longer term, we want more slicing logic to be pushed down
    #       into cpp.  For example, it would be best to pass libcudf
    #       a single sorted table with group offsets).
    df = df.sort_values(partition_cols)
    if not preserve_index:
        df = df.reset_index(drop=True)
    divisions = df[partition_cols].drop_duplicates()
    splits = df[partition_cols].searchsorted(divisions, side="left")
    splits = splits.tolist() + [len(df[partition_cols])]
    return [
        df.iloc[splits[i] : splits[i + 1]].copy(deep=False)
        for i in range(0, len(splits) - 1)
    ]


def _mkdir_if_not_exists(fs, path):
    if fs._isfilestore() and not fs.exists(path):
        try:
            fs.mkdir(path)
        except OSError:
            assert fs.exists(path)


# Logic chosen to match: https://arrow.apache.org/
# docs/_modules/pyarrow/parquet.html#write_to_dataset
def write_to_dataset(
    df, root_path, partition_cols=None, fs=None, preserve_index=False, **kwargs
):
    """Wraps `to_parquet` to write partitioned Parquet datasets.
    For each combination of partition group and value,
    subdirectories are created as follows:

    root_dir/
      group=value1
        <uuid>.parquet
      ...
      group=valueN
        <uuid>.parquet

    Parameters
    ----------
    df : cudf.DataFrame
    root_path : string,
        The root directory of the dataset
    fs : FileSystem, default None
        If nothing passed, paths assumed to be found in the local on-disk
        filesystem
    preserve_index : bool, default False
        Preserve index values in each parquet file.
    partition_cols : list,
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given
    **kwargs : dict,
        kwargs for to_parquet function.
    """

    fs, root_path = pq._get_filesystem_and_path(fs, root_path)
    _mkdir_if_not_exists(fs, root_path)

    if partition_cols is not None and len(partition_cols) > 0:

        data_cols = df.columns.drop(partition_cols)
        if len(data_cols) == 0:
            raise ValueError("No data left to save outside partition columns")

        #  Loop through the partition groups
        for i, sub_df in enumerate(
            _get_partition_groups(
                df, partition_cols, preserve_index=preserve_index
            )
        ):
            if sub_df is None or len(sub_df) == 0:
                continue
            keys = tuple([sub_df[col].iloc[0] for col in partition_cols])
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
            write_df = sub_df.copy(deep=False)
            write_df.drop(columns=partition_cols, inplace=True)
            write_df.to_parquet(full_path, index=preserve_index, **kwargs)
    else:
        outfile = guid() + ".parquet"
        full_path = "/".join([root_path, outfile])
        df.to_parquet(full_path, index=preserve_index, **kwargs)


@ioutils.doc_read_parquet_metadata()
def read_parquet_metadata(path):
    """{docstring}"""

    pq_file = pq.ParquetFile(path)

    num_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    col_names = pq_file.schema.names

    return num_rows, num_row_groups, col_names


@ioutils.doc_read_parquet()
def read_parquet(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    row_group=None,
    row_group_count=None,
    skip_rows=None,
    num_rows=None,
    strings_to_categorical=False,
    use_pandas_metadata=True,
    *args,
    **kwargs,
):
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, None, **kwargs
    )
    if compression is not None:
        raise ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        return libparquet.read_parquet(
            filepath_or_buffer,
            columns=columns,
            row_group=row_group,
            row_group_count=row_group_count,
            skip_rows=skip_rows,
            num_rows=num_rows,
            strings_to_categorical=strings_to_categorical,
            use_pandas_metadata=use_pandas_metadata,
        )
    else:
        warnings.warn("Using CPU via PyArrow to read Parquet dataset.")
        pa_table = pq.read_pandas(
            filepath_or_buffer, columns=columns, *args, **kwargs
        )
        return cudf.DataFrame.from_arrow(pa_table)


@ioutils.doc_to_parquet()
def to_parquet(
    df,
    path,
    engine="cudf",
    compression="snappy",
    index=None,
    partition_cols=None,
    statistics="ROWGROUP",
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "cudf":
        if partition_cols:
            write_to_dataset(
                df,
                path,
                partition_cols=partition_cols,
                preserve_index=index,
                **kwargs,
            )
            return

        # Ensure that no columns dtype is 'category'
        for col in df.columns:
            if df[col].dtype.name == "category":
                raise ValueError(
                    "'category' column dtypes are currently not "
                    + "supported by the gpu accelerated parquet writer"
                )

        return libparquet.write_parquet(
            df, path, index, compression=compression, statistics=statistics
        )
    else:

        # If index is empty set it to the expected default value of True
        if index is None:
            index = True

        pa_table = df.to_arrow(preserve_index=index)
        pq.write_to_dataset(
            pa_table, path, partition_cols=partition_cols, *args, **kwargs
        )
