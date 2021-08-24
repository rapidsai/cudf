# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import warnings
from collections import defaultdict

from pyarrow import dataset as ds, parquet as pq

import cudf
from cudf._lib import parquet as libparquet
from cudf.io.utils import write_to_dataset
from cudf.utils import ioutils
from cudf.utils.dtypes import is_list_like


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
    filters=None,
    row_groups=None,
    skiprows=None,
    num_rows=None,
    strings_to_categorical=False,
    use_pandas_metadata=True,
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

    filepaths_or_buffers = []
    for source in filepath_or_buffer:
        if ioutils.is_directory(source, **kwargs):
            fs = ioutils._ensure_filesystem(
                passed_filesystem=None, path=source
            )
            source = ioutils.stringify_pathlike(source)
            source = fs.sep.join([source, "*.parquet"])

        tmp_source, compression = ioutils.get_filepath_or_buffer(
            path_or_data=source, compression=None, **kwargs,
        )
        if compression is not None:
            raise ValueError(
                "URL content-encoding decompression is not supported"
            )
        if isinstance(tmp_source, list):
            filepath_or_buffer.extend(tmp_source)
        else:
            filepaths_or_buffers.append(tmp_source)

    if columns is not None:
        if not is_list_like(columns):
            raise ValueError("Expected list like for columns")

    if filters is not None:
        # Convert filters to ds.Expression
        filters = pq._filters_to_expression(filters)

        # Initialize ds.FilesystemDataset
        dataset = ds.dataset(
            filepaths_or_buffers, format="parquet", partitioning="hive"
        )

        # Load IDs of filtered row groups for each file in dataset
        filtered_rg_ids = defaultdict(list)
        for fragment in dataset.get_fragments(filter=filters):
            for rg_fragment in fragment.split_by_row_group(filters):
                for rg_info in rg_fragment.row_groups:
                    filtered_rg_ids[rg_fragment.path].append(rg_info.id)

        # Initialize row_groups to be selected
        if row_groups is None:
            row_groups = [None for _ in dataset.files]

        # Store IDs of selected row groups for each file
        for i, file in enumerate(dataset.files):
            if row_groups[i] is None:
                row_groups[i] = filtered_rg_ids[file]
            else:
                row_groups[i] = filter(
                    lambda id: id in row_groups[i], filtered_rg_ids[file]
                )

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
        warnings.warn("Using CPU via PyArrow to read Parquet dataset.")
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
    statistics="ROWGROUP",
    metadata_file_path=None,
    int96_timestamps=False,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "cudf":
        if partition_cols:
            write_to_dataset(
                df,
                filename=partition_file_name,
                partition_cols=partition_cols,
                root_path=path,
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

        path_or_buf = ioutils.get_writer_filepath_or_buffer(
            path, mode="wb", **kwargs
        )
        if ioutils.is_fsspec_open_file(path_or_buf):
            with path_or_buf as file_obj:
                file_obj = ioutils.get_IOBase_writer(file_obj)
                write_parquet_res = libparquet.write_parquet(
                    df,
                    path=file_obj,
                    index=index,
                    compression=compression,
                    statistics=statistics,
                    metadata_file_path=metadata_file_path,
                    int96_timestamps=int96_timestamps,
                )
        else:
            write_parquet_res = libparquet.write_parquet(
                df,
                path=path_or_buf,
                index=index,
                compression=compression,
                statistics=statistics,
                metadata_file_path=metadata_file_path,
                int96_timestamps=int96_timestamps,
            )

        return write_parquet_res

    else:

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


ParquetWriter = libparquet.ParquetWriter
