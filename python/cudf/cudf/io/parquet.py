# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import io
import json
import warnings
from collections import defaultdict
from contextlib import ExitStack
from typing import Dict, List, Tuple
from uuid import uuid4

import fsspec
import numpy as np
from pyarrow import dataset as ds, parquet as pq

import cudf
from cudf._lib import parquet as libparquet
from cudf.api.types import is_list_like
from cudf.core.column import as_column, build_categorical_column
from cudf.utils import ioutils


def _write_parquet(
    df,
    paths,
    compression="snappy",
    index=None,
    statistics="ROWGROUP",
    metadata_file_path=None,
    int96_timestamps=False,
    row_group_size_bytes=None,
    row_group_size_rows=None,
    partitions_info=None,
    **kwargs,
):
    if is_list_like(paths) and len(paths) > 1:
        if partitions_info is None:
            ValueError("partition info is required for multiple paths")
        elif not is_list_like(partitions_info):
            ValueError("partition info must be list-like for multiple paths")
        elif not len(paths) == len(partitions_info):
            ValueError("partitions_info and paths must be of same size")
    if is_list_like(partitions_info) and len(partitions_info) > 1:
        if not is_list_like(paths):
            ValueError("paths must be list-like when partitions_info provided")

    paths_or_bufs = [
        ioutils.get_writer_filepath_or_buffer(path, mode="wb", **kwargs)
        for path in paths
    ]
    common_args = {
        "index": index,
        "compression": compression,
        "statistics": statistics,
        "metadata_file_path": metadata_file_path,
        "int96_timestamps": int96_timestamps,
        "row_group_size_bytes": row_group_size_bytes,
        "row_group_size_rows": row_group_size_rows,
        "partitions_info": partitions_info,
    }
    if all([ioutils.is_fsspec_open_file(buf) for buf in paths_or_bufs]):
        with ExitStack() as stack:
            fsspec_objs = [stack.enter_context(file) for file in paths_or_bufs]
            file_objs = [
                ioutils.get_IOBase_writer(file_obj) for file_obj in fsspec_objs
            ]
            write_parquet_res = libparquet.write_parquet(
                df, filepaths_or_buffers=file_objs, **common_args
            )
    else:
        write_parquet_res = libparquet.write_parquet(
            df, filepaths_or_buffers=paths_or_bufs, **common_args
        )

    return write_parquet_res


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
    **kwargs,
):
    """Wraps `to_parquet` to write partitioned Parquet datasets.
    For each combination of partition group and value,
    subdirectories are created as follows:

    .. code-block:: bash

        root_dir/
            group=value1
                <filename>.parquet
            ...
            group=valueN
                <filename>.parquet

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
    **kwargs : dict,
        kwargs for to_parquet function.
    """

    fs = ioutils._ensure_filesystem(fs, root_path, **kwargs)
    fs.mkdirs(root_path, exist_ok=True)

    if partition_cols is not None and len(partition_cols) > 0:

        (
            full_paths,
            metadata_file_paths,
            grouped_df,
            part_offsets,
            _,
        ) = _get_partitioned(
            df,
            root_path,
            partition_cols,
            filename,
            fs,
            preserve_index,
            **kwargs,
        )

        if return_metadata:
            kwargs["metadata_file_path"] = metadata_file_paths
        metadata = to_parquet(
            grouped_df,
            full_paths,
            index=preserve_index,
            partition_offsets=part_offsets,
            **kwargs,
        )

    else:
        filename = filename or _generate_filename()
        full_path = fs.sep.join([root_path, filename])
        if return_metadata:
            kwargs["metadata_file_path"] = filename
        metadata = df.to_parquet(full_path, index=preserve_index, **kwargs)

    return metadata


@ioutils.doc_read_parquet_metadata()
def read_parquet_metadata(path):
    """{docstring}"""

    pq_file = pq.ParquetFile(path)

    num_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    col_names = pq_file.schema.names

    return num_rows, num_row_groups, col_names


def _process_dataset(
    paths, fs, filters=None, row_groups=None, categorical_partitions=True,
):
    # Returns:
    #     file_list - Expanded/filtered list of paths
    #     row_groups - Filtered list of row-group selections
    #     partition_keys - list of partition keys for each file
    #     partition_categories - Categories for each partition

    # The general purpose of this function is to (1) expand
    # directory input into a list of paths (using the pyarrow
    # dataset API), (2) to apply row-group filters, and (3)
    # to discover directory-partitioning information

    # Deal with case that the user passed in a directory name
    file_list = paths
    if len(paths) == 1 and ioutils.is_directory(paths[0]):
        paths = ioutils.stringify_pathlike(paths[0])

    # Convert filters to ds.Expression
    if filters is not None:
        filters = pq._filters_to_expression(filters)

    # Initialize ds.FilesystemDataset
    dataset = ds.dataset(
        paths, filesystem=fs, format="parquet", partitioning="hive",
    )
    file_list = dataset.files
    if len(file_list) == 0:
        raise FileNotFoundError(f"{paths} could not be resolved to any files")

    # Deal with directory partitioning
    # Get all partition keys (without filters)
    partition_categories = defaultdict(list)
    file_fragment = None
    for file_fragment in dataset.get_fragments():
        keys = ds._get_partition_keys(file_fragment.partition_expression)
        if not (keys or partition_categories):
            # Bail - This is not a directory-partitioned dataset
            break
        for k, v in keys.items():
            if v not in partition_categories[k]:
                partition_categories[k].append(v)
        if not categorical_partitions:
            # Bail - We don't need to discover all categories.
            # We only need to save the partition keys from this
            # first `file_fragment`
            break

    if partition_categories and file_fragment is not None:
        # Check/correct order of `categories` using last file_frag,
        # because `_get_partition_keys` does NOT preserve the
        # partition-hierarchy order of the keys.
        cat_keys = [
            part.split("=")[0]
            for part in file_fragment.path.split(fs.sep)
            if "=" in part
        ]
        if set(partition_categories) == set(cat_keys):
            partition_categories = {
                k: partition_categories[k]
                for k in cat_keys
                if k in partition_categories
            }

    # If we do not have partitioned data and
    # are not filtering, we can return here
    if filters is None and not partition_categories:
        return file_list, row_groups, [], {}

    # Record initial row_groups input
    row_groups_map = {}
    if row_groups is not None:
        # Make sure paths and row_groups map 1:1
        # and save the initial mapping
        if len(paths) != len(file_list):
            raise ValueError(
                "Cannot specify a row_group selection for a directory path."
            )
        row_groups_map = {path: rgs for path, rgs in zip(paths, row_groups)}

    # Apply filters and discover partition columns
    partition_keys = []
    if partition_categories or filters is not None:
        file_list = []
        if filters is not None:
            row_groups = []
        for file_fragment in dataset.get_fragments(filter=filters):
            path = file_fragment.path

            # Extract hive-partition keys, and make sure they
            # are orederd the same as they are in `partition_categories`
            if partition_categories:
                raw_keys = ds._get_partition_keys(
                    file_fragment.partition_expression
                )
                partition_keys.append(
                    [
                        (name, raw_keys[name])
                        for name in partition_categories.keys()
                    ]
                )

            # Apply row-group filtering
            selection = row_groups_map.get(path, None)
            if selection is not None or filters is not None:
                filtered_row_groups = [
                    rg_info.id
                    for rg_fragment in file_fragment.split_by_row_group(
                        filters, schema=dataset.schema,
                    )
                    for rg_info in rg_fragment.row_groups
                ]
            file_list.append(path)
            if filters is not None:
                if selection is None:
                    row_groups.append(filtered_row_groups)
                else:
                    row_groups.append(
                        [
                            rg_id
                            for rg_id in filtered_row_groups
                            if rg_id in selection
                        ]
                    )

    return (
        file_list,
        row_groups,
        partition_keys,
        partition_categories if categorical_partitions else {},
    )


def _get_byte_ranges(file_list, row_groups, columns, fs, **kwargs):

    # This utility is used to collect the footer metadata
    # from a parquet file. This metadata is used to define
    # the exact byte-ranges that will be needed to read the
    # target column-chunks from the file.
    #
    # This utility is only used for remote storage.
    #
    # The calculated byte-range information is used within
    # cudf.io.ioutils.get_filepath_or_buffer (which uses
    # _fsspec_data_transfer to convert non-local fsspec file
    # objects into local byte buffers).

    if row_groups is None:
        if columns is None:
            return None, None, None  # No reason to construct this
        row_groups = [None for path in file_list]

    # Construct a list of required byte-ranges for every file
    all_byte_ranges, all_footers, all_sizes = [], [], []
    for path, rgs in zip(file_list, row_groups):

        # Step 0 - Get size of file
        if fs is None:
            file_size = path.size
        else:
            file_size = fs.size(path)

        # Step 1 - Get 32 KB from tail of file.
        #
        # This "sample size" can be tunable, but should
        # always be >= 8 bytes (so we can read the footer size)
        tail_size = min(kwargs.get("footer_sample_size", 32_000), file_size,)
        if fs is None:
            path.seek(file_size - tail_size)
            footer_sample = path.read(tail_size)
        else:
            footer_sample = fs.tail(path, tail_size)

        # Step 2 - Read the footer size and re-read a larger
        #          tail if necessary
        footer_size = int.from_bytes(footer_sample[-8:-4], "little")
        if tail_size < (footer_size + 8):
            if fs is None:
                path.seek(file_size - (footer_size + 8))
                footer_sample = path.read(footer_size + 8)
            else:
                footer_sample = fs.tail(path, footer_size + 8)

        # Step 3 - Collect required byte ranges
        byte_ranges = []
        md = pq.ParquetFile(io.BytesIO(footer_sample)).metadata
        column_set = None if columns is None else set(columns)
        if column_set is not None:
            schema = md.schema.to_arrow_schema()
            has_pandas_metadata = (
                schema.metadata is not None and b"pandas" in schema.metadata
            )
            if has_pandas_metadata:
                md_index = [
                    ind
                    for ind in json.loads(
                        schema.metadata[b"pandas"].decode("utf8")
                    ).get("index_columns", [])
                    # Ignore RangeIndex information
                    if not isinstance(ind, dict)
                ]
                column_set |= set(md_index)
        for r in range(md.num_row_groups):
            # Skip this row-group if we are targetting
            # specific row-groups
            if rgs is None or r in rgs:
                row_group = md.row_group(r)
                for c in range(row_group.num_columns):
                    column = row_group.column(c)
                    name = column.path_in_schema
                    # Skip this column if we are targetting a
                    # specific columns
                    split_name = name.split(".")[0]
                    if (
                        column_set is None
                        or name in column_set
                        or split_name in column_set
                    ):
                        file_offset0 = column.dictionary_page_offset
                        if file_offset0 is None:
                            file_offset0 = column.data_page_offset
                        num_bytes = column.total_compressed_size
                        byte_ranges.append((file_offset0, num_bytes))

        all_byte_ranges.append(byte_ranges)
        all_footers.append(footer_sample)
        all_sizes.append(file_size)
    return all_byte_ranges, all_footers, all_sizes


@ioutils.doc_read_parquet()
def read_parquet(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    filters=None,
    row_groups=None,
    skiprows=None,
    num_rows=None,
    strings_to_categorical=False,
    use_pandas_metadata=True,
    use_python_file_object=False,
    categorical_partitions=True,
    *args,
    **kwargs,
):
    """{docstring}"""

    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # a list of row groups per source should be passed. make the list of
    # lists that is expected for multiple sources
    if row_groups is not None:
        if not is_list_like(row_groups):
            row_groups = [[row_groups]]
        elif not is_list_like(row_groups[0]):
            row_groups = [row_groups]

    # Check columns input
    if columns is not None:
        if not is_list_like(columns):
            raise ValueError("Expected list like for columns")

    # Start by trying construct a filesystem object, so we
    # can apply filters on remote file-systems
    fs, paths = ioutils._get_filesystem_and_paths(filepath_or_buffer, **kwargs)

    # Use pyarrow dataset to detect/process directory-partitioned
    # data and apply filters. Note that we can only support partitioned
    # data and filtering if the input is a single directory or list of
    # paths.
    partition_keys = []
    partition_categories = {}
    if fs and paths:
        (
            paths,
            row_groups,
            partition_keys,
            partition_categories,
        ) = _process_dataset(
            paths,
            fs,
            filters=filters,
            row_groups=row_groups,
            categorical_partitions=categorical_partitions,
        )
    elif filters is not None:
        raise ValueError("cudf cannot apply filters to open file objects.")
    filepath_or_buffer = paths if paths else filepath_or_buffer

    # Check if we should calculate the specific byte-ranges
    # needed for each parquet file. We always do this when we
    # have a file-system object to work with and it is not a
    # local filesystem object. We can also do it without a
    # file-system object for `AbstractBufferedFile` buffers
    byte_ranges, footers, file_sizes = None, None, None
    if not use_python_file_object:
        need_byte_ranges = fs is not None and not ioutils._is_local_filesystem(
            fs
        )
        if need_byte_ranges or (
            filepath_or_buffer
            and isinstance(
                filepath_or_buffer[0], fsspec.spec.AbstractBufferedFile,
            )
        ):
            byte_ranges, footers, file_sizes = _get_byte_ranges(
                filepath_or_buffer, row_groups, columns, fs, **kwargs
            )

    filepaths_or_buffers = []
    for i, source in enumerate(filepath_or_buffer):

        tmp_source, compression = ioutils.get_filepath_or_buffer(
            path_or_data=source,
            compression=None,
            fs=fs,
            byte_ranges=byte_ranges[i] if byte_ranges else None,
            footer=footers[i] if footers else None,
            file_size=file_sizes[i] if file_sizes else None,
            add_par1_magic=True,
            use_python_file_object=use_python_file_object,
            **kwargs,
        )

        if compression is not None:
            raise ValueError(
                "URL content-encoding decompression is not supported"
            )
        if isinstance(tmp_source, list):
            filepath_or_buffer.extend(tmp_source)
        else:
            filepaths_or_buffers.append(tmp_source)

    # Warn user if they are not using cudf for IO
    # (There is a good chance this was not the intention)
    if engine != "cudf":
        warnings.warn(
            "Using CPU via PyArrow to read Parquet dataset."
            "This option is both inefficient and unstable!"
        )
        if filters is not None:
            warnings.warn(
                "Parquet row-group filtering is only supported with "
                "'engine=cudf'. Use pandas or pyarrow API directly "
                "for full CPU-based filtering functionality."
            )

    return _parquet_to_frame(
        filepaths_or_buffers,
        engine,
        *args,
        columns=columns,
        row_groups=row_groups,
        skiprows=skiprows,
        num_rows=num_rows,
        strings_to_categorical=strings_to_categorical,
        use_pandas_metadata=use_pandas_metadata,
        partition_keys=partition_keys,
        partition_categories=partition_categories,
        **kwargs,
    )


def _parquet_to_frame(
    paths_or_buffers,
    *args,
    row_groups=None,
    partition_keys=None,
    partition_categories=None,
    **kwargs,
):

    # If this is not a partitioned read, only need
    # one call to `_read_parquet`
    if not partition_keys:
        return _read_parquet(
            paths_or_buffers, *args, row_groups=row_groups, **kwargs,
        )

    # For partitioned data, we need a distinct read for each
    # unique set of partition keys. Therefore, we start by
    # aggregating all paths with matching keys using a dict
    plan = {}
    for i, (keys, path) in enumerate(zip(partition_keys, paths_or_buffers)):
        rgs = row_groups[i] if row_groups else None
        tkeys = tuple(keys)
        if tkeys in plan:
            plan[tkeys][0].append(path)
            if rgs is not None:
                plan[tkeys][1].append(rgs)
        else:
            plan[tkeys] = ([path], None if rgs is None else [rgs])

    dfs = []
    for part_key, (key_paths, key_row_groups) in plan.items():
        # Add new DataFrame to our list
        dfs.append(
            _read_parquet(
                key_paths, *args, row_groups=key_row_groups, **kwargs,
            )
        )
        # Add partition columns to the last DataFrame
        for (name, value) in part_key:
            if partition_categories and name in partition_categories:
                # Build the categorical column from `codes`
                codes = as_column(
                    partition_categories[name].index(value),
                    length=len(dfs[-1]),
                )
                dfs[-1][name] = build_categorical_column(
                    categories=partition_categories[name],
                    codes=codes,
                    size=codes.size,
                    offset=codes.offset,
                    ordered=False,
                )
            else:
                # Not building categorical columns, so
                # `value` is already what we want
                dfs[-1][name] = as_column(value, length=len(dfs[-1]))

    # Concatenate dfs and return.
    # Assume we can ignore the index if it has no name.
    return (
        cudf.concat(dfs, ignore_index=dfs[-1].index.name is None)
        if len(dfs) > 1
        else dfs[0]
    )


def _read_parquet(
    filepaths_or_buffers,
    engine,
    columns=None,
    row_groups=None,
    skiprows=None,
    num_rows=None,
    strings_to_categorical=None,
    use_pandas_metadata=None,
    *args,
    **kwargs,
):
    # Simple helper function to dispatch between
    # cudf and pyarrow to read parquet data
    if engine == "cudf":
        return libparquet.read_parquet(
            filepaths_or_buffers,
            columns=columns,
            row_groups=row_groups,
            skiprows=skiprows,
            num_rows=num_rows,
            strings_to_categorical=strings_to_categorical,
            use_pandas_metadata=use_pandas_metadata,
        )
    else:
        return cudf.DataFrame.from_arrow(
            pq.ParquetDataset(filepaths_or_buffers).read_pandas(
                columns=columns, *args, **kwargs
            )
        )


@ioutils.doc_to_parquet()
def to_parquet(
    df,
    path,
    engine="cudf",
    compression="snappy",
    index=None,
    partition_cols=None,
    partition_file_name=None,
    partition_offsets=None,
    statistics="ROWGROUP",
    metadata_file_path=None,
    int96_timestamps=False,
    row_group_size_bytes=None,
    row_group_size_rows=None,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "cudf":
        # Ensure that no columns dtype is 'category'
        for col in df.columns:
            if partition_cols is None or col not in partition_cols:
                if df[col].dtype.name == "category":
                    raise ValueError(
                        "'category' column dtypes are currently not "
                        + "supported by the gpu accelerated parquet writer"
                    )

        if partition_cols:
            if metadata_file_path is not None:
                warnings.warn(
                    "metadata_file_path will be ignored/overwritten when "
                    "partition_cols are provided. To request returning the "
                    "metadata binary blob, pass `return_metadata=True`"
                )
            kwargs.update(
                {
                    "compression": compression,
                    "statistics": statistics,
                    "int96_timestamps": int96_timestamps,
                    "row_group_size_bytes": row_group_size_bytes,
                    "row_group_size_rows": row_group_size_rows,
                }
            )
            return write_to_dataset(
                df,
                filename=partition_file_name,
                partition_cols=partition_cols,
                root_path=path,
                preserve_index=index,
                **kwargs,
            )

        if partition_offsets:
            kwargs["partitions_info"] = list(
                zip(
                    partition_offsets,
                    np.roll(partition_offsets, -1) - partition_offsets,
                )
            )[:-1]

        return _write_parquet(
            df,
            paths=path if is_list_like(path) else [path],
            compression=compression,
            index=index,
            statistics=statistics,
            metadata_file_path=metadata_file_path,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            **kwargs,
        )

    else:
        if partition_offsets is not None:
            warnings.warn(
                "partition_offsets will be ignored when engine is not cudf"
            )

        # If index is empty set it to the expected default value of True
        if index is None:
            index = True

        # Convert partition_file_name to a call back
        if partition_file_name:
            partition_file_name = lambda x: partition_file_name  # noqa: E731

        pa_table = df.to_arrow(preserve_index=index)
        return pq.write_to_dataset(
            pa_table,
            root_path=path,
            partition_filename_cb=partition_file_name,
            partition_cols=partition_cols,
            *args,
            **kwargs,
        )


@ioutils.doc_merge_parquet_filemetadata()
def merge_parquet_filemetadata(filemetadata_list):
    """{docstring}"""

    return libparquet.merge_filemetadata(filemetadata_list)


def _generate_filename():
    return uuid4().hex + ".parquet"


def _get_partitioned(
    df,
    root_path,
    partition_cols,
    filename=None,
    fs=None,
    preserve_index=False,
    **kwargs,
):
    fs = ioutils._ensure_filesystem(fs, root_path, **kwargs)
    fs.mkdirs(root_path, exist_ok=True)
    if not (set(df._data) - set(partition_cols)):
        raise ValueError("No data left to save outside partition columns")

    part_names, part_offsets, _, grouped_df = df.groupby(
        partition_cols
    )._grouped()
    if not preserve_index:
        grouped_df.reset_index(drop=True, inplace=True)
    grouped_df.drop(columns=partition_cols, inplace=True)
    # Copy the entire keys df in one operation rather than using iloc
    part_names = part_names.to_pandas().to_frame(index=False)

    full_paths = []
    metadata_file_paths = []
    for keys in part_names.itertuples(index=False):
        subdir = fs.sep.join(
            [f"{name}={val}" for name, val in zip(partition_cols, keys)]
        )
        prefix = fs.sep.join([root_path, subdir])
        fs.mkdirs(prefix, exist_ok=True)
        filename = filename or _generate_filename()
        full_path = fs.sep.join([prefix, filename])
        full_paths.append(full_path)
        metadata_file_paths.append(fs.sep.join([subdir, filename]))

    return full_paths, metadata_file_paths, grouped_df, part_offsets, filename


ParquetWriter = libparquet.ParquetWriter


class ParquetDatasetWriter:
    def __init__(
        self,
        path,
        partition_cols,
        index=None,
        compression=None,
        statistics="ROWGROUP",
    ) -> None:
        """
        Write a parquet file or dataset incrementally

        Parameters
        ----------
        path : str
            File path or Root Directory path. Will be used as Root Directory
            path while writing a partitioned dataset.
        partition_cols : list
            Column names by which to partition the dataset
            Columns are partitioned in the order they are given
        index : bool, default None
            If ``True``, include the dataframe’s index(es) in the file output.
            If ``False``, they will not be written to the file. If ``None``,
            index(es) other than RangeIndex will be saved as columns.
        compression : {'snappy', None}, default 'snappy'
            Name of the compression to use. Use ``None`` for no compression.
        statistics : {'ROWGROUP', 'PAGE', 'NONE'}, default 'ROWGROUP'
            Level at which column statistics should be included in file.


        Examples
        ________
        Using a context

        >>> df1 = cudf.DataFrame({"a": [1, 1, 2, 2, 1], "b": [9, 8, 7, 6, 5]})
        >>> df2 = cudf.DataFrame({"a": [1, 3, 3, 1, 3], "b": [4, 3, 2, 1, 0]})
        >>> with ParquetDatasetWriter("./dataset", partition_cols=["a"]) as cw:
        ...     cw.write_table(df1)
        ...     cw.write_table(df2)

        By manually calling ``close()``

        >>> cw = ParquetDatasetWriter("./dataset", partition_cols=["a"])
        >>> cw.write_table(df1)
        >>> cw.write_table(df2)
        >>> cw.close()

        Both the methods will generate the same directory structure

        .. code-block:: bash

            dataset/
                a=1
                    <filename>.parquet
                a=2
                    <filename>.parquet
                a=3
                    <filename>.parquet

        """
        self.path = path
        self.common_args = {
            "index": index,
            "compression": compression,
            "statistics": statistics,
        }
        self.partition_cols = partition_cols
        # Collection of `ParquetWriter`s, and the corresponding
        # partition_col values they're responsible for
        self._chunked_writers: List[
            Tuple[libparquet.ParquetWriter, List[str], str]
        ] = []
        # Map of partition_col values to their ParquetWriter's index
        # in self._chunked_writers for reverse lookup
        self.path_cw_map: Dict[str, int] = {}
        self.filename = None

    def write_table(self, df):
        """
        Write a dataframe to the file/dataset
        """
        (
            paths,
            metadata_file_paths,
            grouped_df,
            offsets,
            self.filename,
        ) = _get_partitioned(
            df,
            self.path,
            self.partition_cols,
            preserve_index=self.common_args["index"],
            filename=self.filename,
        )

        existing_cw_batch = defaultdict(dict)
        new_cw_paths = []

        for path, part_info, meta_path in zip(
            paths,
            zip(offsets, np.roll(offsets, -1) - offsets),
            metadata_file_paths,
        ):
            if path in self.path_cw_map:  # path is a currently open file
                cw_idx = self.path_cw_map[path]
                existing_cw_batch[cw_idx][path] = part_info
            else:  # path not currently handled by any chunked writer
                new_cw_paths.append((path, part_info, meta_path))

        # Write out the parts of grouped_df currently handled by existing cw's
        for cw_idx, path_to_part_info_map in existing_cw_batch.items():
            cw = self._chunked_writers[cw_idx][0]
            # match found paths with this cw's paths and nullify partition info
            # for partition_col values not in this batch
            this_cw_part_info = [
                path_to_part_info_map.get(path, (0, 0))
                for path in self._chunked_writers[cw_idx][1]
            ]
            cw.write_table(grouped_df, this_cw_part_info)

        # Create new cw for unhandled paths encountered in this write_table
        new_paths, part_info, meta_paths = zip(*new_cw_paths)
        self._chunked_writers.append(
            (
                ParquetWriter(new_paths, **self.common_args),
                new_paths,
                meta_paths,
            )
        )
        new_cw_idx = len(self._chunked_writers) - 1
        self.path_cw_map.update({k: new_cw_idx for k in new_paths})
        self._chunked_writers[-1][0].write_table(grouped_df, part_info)

    def close(self, return_metadata=False):
        """
        Close all open files and optionally return footer metadata as a binary
        blob
        """

        metadata = [
            cw.close(metadata_file_path=meta_path if return_metadata else None)
            for cw, _, meta_path in self._chunked_writers
        ]

        if return_metadata:
            return (
                merge_parquet_filemetadata(metadata)
                if len(metadata) > 1
                else metadata[0]
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
