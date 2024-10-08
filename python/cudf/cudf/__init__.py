# If libcudf was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcudf
except ModuleNotFoundError:
    pass
else:
    libcudf.load_library()
    del libcudf

from ptxcompiler.patch import safe_get_versions
from cudf._lib import strings_udf
