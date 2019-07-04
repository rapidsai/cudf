from glob import glob

from dask.base import tokenize
from dask.bytes import open_files
from dask.delayed import delayed
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

    name = "read-orc-" + tokenize(path, **kwargs)
    dsk = {}
    if "://" in str(path):
        files = open_files(path)
        with files[0] as fn:
            meta = cudf.read_orc(fn, **kwargs)
        parts = [delayed(_read_orc)(f, **kwargs) for f in files]
        return dd.from_delayed(parts, meta=meta)
    else:
        filenames = sorted(glob(str(path)))
        meta = cudf.read_orc(filenames[0], **kwargs)
        dsk = {
            (name, i): (apply, cudf.read_orc, [fn], kwargs)
            for i, fn in enumerate(filenames)
        }

        divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_orc(fn, **kwargs):
    with fn as f:
        return cudf.read_orc(f, **kwargs)
