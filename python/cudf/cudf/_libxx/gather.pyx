from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *
from cudf._libxx.table cimport *

def py_gather(Table source_table, Column gather_map):
    c_result = _Table()
    c_result.c_obj = gather(
            source_table.view(),
            gather_map.view())
    return c_result.release_into_table()



