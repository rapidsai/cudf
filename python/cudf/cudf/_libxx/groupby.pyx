from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

cimport cudf._libxx.cpp.groupby as libcudf_groupby


cdef class GroupBy:
    cdef unique_ptr[libcudf_groupby.groupby] c_obj
    cdef dict __dict__

    def __cinit__(self, Table keys, *args, **kwargs):
        """
        GroupBy object
        """
        self.c_obj.reset(new libcudf_groupby.groupby(keys.data_view()))

    def __init__(self, keys):
        self.keys = keys

    def groups(self, Table values):
        c_groups = move(self.c_obj.get()[0].get_groups(values.data_view()))
        c_grouped_keys = move(c_groups.keys)
        c_grouped_values = move(c_groups.values)
        c_group_offsets = c_groups.offsets

        grouped_keys = Table.from_unique_ptr(
            move(c_grouped_keys),
            self.keys._column_names
        )

        grouped_values = Table.from_unique_ptr(
            move(c_grouped_values),
            values._column_names
        )

        return grouped_keys, grouped_values, c_group_offsets

