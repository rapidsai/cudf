from ._gdf import ffi, libgdf, np_to_gdf_dtype


def _wrap_string(text):
    return ffi.new("char[]", text.encode())


def read_csv(path, names, dtypes, delimiter=',', ):
    out = libgdf.read_csv(
        _wrap_string(path),
        delimiter.encode(),
        len(names),
        [_wrap_string(k) for k in names],
        list(map(lambda x: _wrap_string(str(x)), dtypes)),
    )
    if out == ffi.NULL:
        raise ValueError("Failed to parse CSV")
    return out
