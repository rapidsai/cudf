cimport cudf._libxx.lib as libcudf
from cudf._libxx.column cimport *
from cudf._libxx.table cimport *


def gather(Table source_table, Column gather_map):
    c_result = _Table.from_ptr(
        libcudf.gather(
            source_table.view(),
            gather_map.view()
            )
        )
    return c_result.release_into_table()

