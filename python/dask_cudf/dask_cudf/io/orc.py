from glob import glob

import dask.dataframe as dd
from dask.base import tokenize
from dask.bytes import open_files
from dask.compatibility import apply

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

        # An `OpenFile` should be used in a Context
        with files[0] as f:
            meta = cudf.read_orc(f, **kwargs)

        dsk = {
            (name, i): (apply, _read_orc, [f], kwargs)
            for i, f in enumerate(files)
        }
    else:
        if isinstance(path, list):
            filenames = path
        elif isinstance(path, str):
            filenames = sorted(glob(path))
        elif hasattr(path, "__fspath__"):
            filenames = sorted(glob(path.__fspath__()))
        else:
            raise TypeError("Path type not understood:{}".format(type(path)))

        meta = cudf.read_orc(filenames[0], **kwargs)
        dsk = {
            (name, i): (apply, cudf.read_orc, [fn], kwargs)
            for i, fn in enumerate(filenames)
        }

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_orc(file_obj, **kwargs):
    with file_obj as f:
        return cudf.read_orc(f, **kwargs)
