from cudf._libxx.column cimport *
from cudf._libxx.table cimport *

cimport cudf._libxx.lib as libcudf


def gather(_Table source_table, Column gather_map):
    cdef unique_ptr[table] c_result = (
        libcudf.gather(
            source_table.view(),
            gather_map.view()
        )
    )
    return _Table.from_ptr(move(c_result))
