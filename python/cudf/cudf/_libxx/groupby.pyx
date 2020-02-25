from libcpp.pair cimport pair
from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table
cimport cudf._libxx.includes.groupby as libgroupby

cdef class GroupBy:
    cdef unique_ptr[libgroupby.groupby] c_obj
    cdef dict __dict__

    def __cinit__(self, Table keys, *args, **kwargs):
        self.keys = keys
        self.c_obj.reset(new libgroupby.groupby(keys.data_view()))

    def groups(self, values=None):
        c_groups = libgroupby.move(self.c_obj.get()[0].get_groups())
        c_grouped_keys = move(c_groups.group_keys)
        c_group_offsets = c_groups.group_offsets
        grouped_keys = self.keys.__class__._from_table(
            Table.from_unique_ptr(move(c_grouped_keys),
                                  self.keys._column_names))
        for i in range(c_group_offsets.size()-1):
            yield grouped_keys[c_group_offsets[i]:c_group_offsets[i+1]]
        
