# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class ColumnView:
    """Wrapper around column_view."""
    # TODO: Need a way to map the data buffer size to the number of
    # elements. For fixed width types a mapping could be made based on the
    # number of bytes they occupy, but not for nested types. Not sure how
    # best to expose that in the API yet, but matching C++ for now and
    # requesting the size from the user. The gpumemoryview may also help, if it
    # is typed then it would contain the necessary information (total buffer
    # size and sizeof(type))
    # TODO: I've temporarily defined __init__ instead of __cinit__ so that
    # factory functions can call __new__ without arguments. I'll need to think
    # more fully about what construction patterns we actually want to support.
    # TODO: At the moment libcudf does not expose APIs for counting the nulls
    # in a bitmask directly (those APIs are in detail/null_mask). We'll need to
    # expose those eventually once UNKNOWN_NULL_COUNT goes away. This dovetails
    # with our desire to expose other functionality too like bitmask_and.
    # TODO: The nature of view types in libcudf is antithetical to how Python
    # users expect to interact with anything. The idea that an object could
    # become invalidated and then just seg fault (if the memory owner goes out
    # of scope) is pretty terrible in Python. We need the gpumemoryview to
    # maintain a reference to the owner so that it isn't destroyed to avoid
    # this problem (probably should just have it use __cuda_array_interface__).
    def __init__(
        self, DataType dtype not None, size_type size, gpumemoryview data_buf,
        gpumemoryview mask_buf, size_type null_count, size_type offset,
        # TODO: Not sure what the best input is for children, for now just
        # using a List[ColumnView]
        object children
    ):
        # TODO: Investigate cases where the data_buf is None. I'm not sure that
        # this is a real use case that we should support. EDIT: It looks like
        # this is something that libcudf itself supports, so I guess it's fine
        # but I would still like to better understand when it occurs.
        cdef const void * data = NULL
        if data_buf is not None:
            data = int_to_void_ptr(data_buf.ptr)
        cdef const bitmask_type * null_mask = NULL
        if mask_buf is not None:
            null_mask = int_to_bitmask_ptr(mask_buf.ptr)

        cdef vector[column_view] c_children
        cdef ColumnView child
        if children is not None:
            for child in children:
                c_children.push_back(dereference(child.get()))

        self.c_obj.reset(
            new column_view(
                dtype.c_obj, size, data, null_mask, null_count, offset,
                c_children
            )
        )

    cdef column_view * get(self) nogil:
        """Get the underlying column_view object."""
        return self.c_obj.get()

    @staticmethod
    cdef from_column_view(column_view cv):
        cdef ColumnView ret = ColumnView.__new__(ColumnView)
        ret.c_obj.reset(new column_view(cv))
        return ret

    cpdef size_type size(self):
        return self.get().size()

    cpdef size_type null_count(self):
        return self.get().null_count()
