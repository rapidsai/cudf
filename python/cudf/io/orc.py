# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame

import pyarrow.orc as orc
import warnings


def read_orc(path, columns=None):
    """
    Load an orc object from the file path, returning a DataFrame.

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

      df = cudf.read_orc(filename)

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
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    warnings.warn("Using CPU via PyArrow to read ORC dataset, this will "
                  "be GPU accelerated in the future")
    orc_file = orc.ORCFile(path)
    pa_table = orc_file.read(columns=columns)
    return DataFrame.from_arrow(pa_table)
