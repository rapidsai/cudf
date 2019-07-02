from glob import glob

from dask.base import tokenize
from dask.compatibility import apply
import dask.dataframe as dd

import cudf


def read_orc(path, **kwargs):
    """ Read ORC files into a Dask DataFrame

    This calls the ``cudf.read_orc`` function on many ORC files.
    See that function for additional details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_orc("/path/to/*.orc")  # doctest: +SKIP

    See Also
    --------
    cudf.read_orc
    """

    filenames = sorted(glob(str(path)))
    name = "read-orc-" + tokenize(path, **kwargs)

    meta = cudf.read_orc(filenames[0], **kwargs)

    graph = {
        (name, i): (apply, cudf.read_orc, [fn], kwargs)
        for i, fn in enumerate(filenames)
    }

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
