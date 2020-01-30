# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

import pyarrow.parquet as pq

import cudf
import cudf._lib as libcudf
from cudf.utils import ioutils


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
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        df = libcudf.parquet.read_parquet(
            filepath_or_buffer,
            columns,
            row_group,
            skip_rows,
            num_rows,
            strings_to_categorical,
            use_pandas_metadata,
        )
    else:
        warnings.warn("Using CPU via PyArrow to read Parquet dataset.")
        pa_table = pq.read_pandas(
            filepath_or_buffer, columns=columns, *args, **kwargs
        )
        df = cudf.DataFrame.from_arrow(pa_table)

    return df


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

        if index is not None:
            ValueError(
                "'index' is currently not supported by the gpu "
                + "accelerated parquet writer"
            )

        if partition_cols is not None:
            ValueError(
                "'partition_cols' is currently not supported by the "
                + "gpu accelerated parquet writer"
            )

        # Ensure that no columns dtype is 'category'
        for col in df.columns:
            if df[col].dtype.name == "category":
                ValueError(
                    "'category' column dtypes are currently not "
                    + "supported by the gpu accelerated parquet writer"
                )

        return libcudf.parquet.write_parquet(
            df, path, compression=compression, statistics=statistics
        )
    else:

        # If index is empty set it to the expected default value of True
        if index is None:
            index = True

        pa_table = df.to_arrow(preserve_index=index)
        pq.write_to_dataset(
            pa_table, path, partition_cols=partition_cols, *args, **kwargs,
        )
