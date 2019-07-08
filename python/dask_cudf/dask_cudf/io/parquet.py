from glob import glob

import dask.dataframe as dd
from dask.base import tokenize
from dask.compatibility import apply
from dask.utils import natural_sort_key

import cudf


def read_parquet(path, **kwargs):
    """ Read parquet files into a Dask DataFrame

    This calls the ``cudf.read_parquet`` function on many parquet files.
    See that function for additional details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_parquet("/path/to/dataset/")  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """

    name = "read-parquet-" + tokenize(path, **kwargs)

    paths = path
    if isinstance(path, str):
        paths = sorted(glob(str(path)))

    # Ignore *_metadata files for now
    paths = sorted(
        [f for f in paths if not f.endswith("_metadata")], key=natural_sort_key
    )

    # Use 0th file to create meta
    meta = cudf.read_parquet(paths[0], **kwargs)
    graph = {
        (name, i): (apply, cudf.read_parquet, [fn], kwargs)
        for i, fn in enumerate(paths)
    }
    divisions = [None] * (len(paths) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
