# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame

from pyarrow import feather
import warnings


def read_feather(path, columns=None, **kwargs):
    """
    Load an feather object from the file path, returning a DataFrame.

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

      df = cudf.read_feather(filename)

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
    .to_feather
    """

    warnings.warn("Using CPU via PyArrow to read feather dataset, this may "
                  "be GPU accelerated in the future")
    pa_table = feather.read_table(path, columns=columns, **kwargs)
    return DataFrame.from_arrow(pa_table)


def to_feather(df, path, **kwargs):
    """
    Write a DataFrame to the feather format.
    Parameters
    ----------
    path : str
        File path

    See Also
    --------
    .read_feather
    """
    warnings.warn("Using CPU via PyArrow to write Feather dataset, this may "
                  "be GPU accelerated in the future")
    # Feather doesn't support using an index
    pa_table = df.to_arrow(preserve_index=False)
    feather_writer = feather.FeatherWriter(path)
    for i, name in enumerate(pa_table.schema.names):
        col = pa_table[i]
        feather.check_chunked_overflow(col)
        feather_writer.writer.write_array(name, col.data.chunk(0))
    feather_writer.writer.close()
