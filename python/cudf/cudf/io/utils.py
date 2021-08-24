# Copyright (c) 2021, NVIDIA CORPORATION.

from uuid import uuid4

from cudf.utils import ioutils


def _get_partition_groups(df, partition_cols, preserve_index=False):
    # TODO: We can use groupby functionality here after cudf#4346.
    #       Longer term, we want more slicing logic to be pushed down
    #       into cpp.  For example, it would be best to pass libcudf
    #       a single sorted table with group offsets).
    df = df.sort_values(partition_cols)
    if not preserve_index:
        df = df.reset_index(drop=True)
    divisions = df[partition_cols].drop_duplicates(ignore_index=True)
    splits = df[partition_cols].searchsorted(divisions, side="left")
    splits = splits.tolist() + [len(df[partition_cols])]
    return [
        df.iloc[splits[i] : splits[i + 1]].copy(deep=False)
        for i in range(0, len(splits) - 1)
    ]


# Logic chosen to match: https://arrow.apache.org/
# docs/_modules/pyarrow/parquet.html#write_to_dataset
def write_to_dataset(
    df,
    root_path,
    filename=None,
    partition_cols=None,
    fs=None,
    preserve_index=False,
    return_metadata=False,
    file_format="parquet",
    **kwargs,
):
    """Wraps `to_parquet` and `to_orc` to write partitioned datasets.
    For each combination of partition group and value, subdirectories
    are created as follows:

    .. code-block:: bash

        root_dir/
            group=value1
                <filename>
            ...
            group=valueN
                <filename>

    Parameters
    ----------
    df : cudf.DataFrame
    root_path : string,
        The root directory of the dataset
    filename : string, default None
        The file name to use (within each partition directory). If None,
        a random uuid4 hex string will be used for each file name.
    fs : FileSystem, default None
        If nothing passed, paths assumed to be found in the local on-disk
        filesystem
    preserve_index : bool, default False
        Preserve index values in each parquet file.
    partition_cols : list,
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given
    return_metadata : bool, default False
        Return parquet metadata for written data. Returned metadata will
        include the file-path metadata (relative to `root_path`).
    file_format : "parquet" or "orc", default "parquet"
        Output file format.
    **kwargs : dict,
        kwargs for to_parquet function.
    """

    fs = ioutils._ensure_filesystem(fs, root_path)
    fs.mkdirs(root_path, exist_ok=True)
    metadata = []

    # Check file format
    use_kwargs = kwargs.copy()
    if file_format == "orc":
        from .orc import to_orc as io_func

        if return_metadata:
            raise ValueError(
                "return_metadata=True not supported for file_format='orc'"
            )
        if preserve_index:
            raise ValueError(
                "preserve_index=True not supported for file_format='orc'"
            )
    elif file_format == "parquet":
        from .parquet import merge_parquet_filemetadata, to_parquet as io_func

        use_kwargs["index"] = preserve_index
    else:
        raise ValueError(
            f"file_format={file_format} not recognized. Use 'parquet' or 'orc'"
        )

    if partition_cols is not None and len(partition_cols) > 0:

        data_cols = df.columns.drop(partition_cols)
        if len(data_cols) == 0:
            raise ValueError("No data left to save outside partition columns")

        #  Loop through the partition groups
        for _, sub_df in enumerate(
            _get_partition_groups(
                df, partition_cols, preserve_index=preserve_index
            )
        ):
            if sub_df is None or len(sub_df) == 0:
                continue
            keys = tuple([sub_df[col].iloc[0] for col in partition_cols])
            if not isinstance(keys, tuple):
                keys = (keys,)
            subdir = fs.sep.join(
                [
                    "{colname}={value}".format(colname=name, value=val)
                    for name, val in zip(partition_cols, keys)
                ]
            )
            prefix = fs.sep.join([root_path, subdir])
            fs.mkdirs(prefix, exist_ok=True)
            filename = filename or uuid4().hex + f".{file_format}"
            full_path = fs.sep.join([prefix, filename])
            write_df = sub_df.copy(deep=False)
            write_df.drop(columns=partition_cols, inplace=True)
            with fs.open(full_path, mode="wb") as fil:
                fil = ioutils.get_IOBase_writer(fil)
                if return_metadata:
                    metadata.append(
                        io_func(
                            write_df,
                            fil,
                            metadata_file_path=fs.sep.join([subdir, filename]),
                            **use_kwargs,
                        )
                    )
                else:
                    io_func(write_df, fil, **use_kwargs)

    else:
        filename = filename or uuid4().hex + f".{file_format}"
        full_path = fs.sep.join([root_path, filename])
        if return_metadata:
            metadata.append(
                io_func(
                    df, full_path, metadata_file_path=filename, **use_kwargs,
                )
            )
        else:
            io_func(df, full_path, **use_kwargs)

    if metadata:
        return (
            merge_parquet_filemetadata(metadata)
            if len(metadata) > 1
            else metadata[0]
        )
