import cudf
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from cudf._lib.table cimport Table
from libcpp.string cimport string

cdef extern from "src/kernel_wrapper.hh":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl)
        void tenth_mm_to_inches(int column_index)
        void mm_to_inches(int column_index)

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf

    def __cinit__(self, Table t):
        self.gdf = new C_CudfWrapper(t.mutable_view())

    def tenth_mm_to_inches(self, col_index):
        self.gdf.tenth_mm_to_inches(col_index)

    def mm_to_inches(self, col_index):
        self.gdf.mm_to_inches(col_index)
