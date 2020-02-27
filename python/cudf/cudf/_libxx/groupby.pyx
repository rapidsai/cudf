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
        self.keys = keys
        self.c_obj.reset(new libcudf_groupby.groupby(keys.data_view()))

    def groups(self, values=None):
        c_groups = move(self.c_obj.get()[0].get_groups())
        c_grouped_keys = move(c_groups.keys)
        c_group_offsets = c_groups.offsets
        grouped_keys = self.keys.__class__._from_table(
            Table.from_unique_ptr(move(c_grouped_keys),
                                  self.keys._column_meta))
        for i in range(c_group_offsets.size()-1):
            yield grouped_keys[c_group_offsets[i]:c_group_offsets[i+1]]
