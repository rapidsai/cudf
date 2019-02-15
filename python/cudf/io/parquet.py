# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame

import pyarrow.parquet as pq
import warnings


def read_parquet(path, *args, **kwargs):
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : string
        File path
    columns : list, default=None
        If not None, only these columns will be read from the file.

    Returns
    -------
    DataFrame

    Examples
    --------


    .. code-block:: python

      import cudf

      df = cudf.read_parquet(filename)

      # Display results
      print(df)

    Output:

    .. code-block:: python

          num1                datetime text
        0  123 2018-11-13T12:00:00.000 5451
        1  456 2018-11-14T12:35:01.000 5784
        2  789 2018-11-15T18:02:59.000 6117

    See Also
    --------
    cudf.io.parquet.to_parquet
    cudf.io.orc.read_orc
    """

    warnings.warn("Using CPU via PyArrow to read Parquet dataset, this will "
                  "be GPU accelerated in the future")
    pa_table = pq.read_pandas(path, *args, **kwargs)
    return DataFrame.from_arrow(pa_table)


def to_parquet(df, path, *args, **kwargs):
    """
    Write a DataFrame to the parquet format.
    Parameters
    ----------
    path : str
        File path or Root Directory path. Will be used as Root Directory path
        while writing a partitioned dataset.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file. If ``None``, the
        engine's default behavior will be used.
    partition_cols : list, optional, default None
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.orc.read_orc
    """
    warnings.warn("Using CPU via PyArrow to write Parquet dataset, this will "
                  "be GPU accelerated in the future")
    pa_table = df.to_arrow()
    pq.write_to_dataset(pa_table, path, *args, **kwargs)
