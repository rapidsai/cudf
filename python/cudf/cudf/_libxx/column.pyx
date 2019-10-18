import numpy as np

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer
from rmm._lib.device_buffer import DeviceBuffer

from cudf._libxx.lib cimport *


np_to_cudf_types = {np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64}

cdef class Column:
    def __cinit__(self):
        pass

    @classmethod
    def from_array(cls, array):
        cdef Column col = Column.__new__(Column)
        cdef type_id dtype = np_to_cudf_types[array.dtype]
        buf = DeviceBuffer(array)
        col.c_obj = new column(
            data_type(dtype),
            len(array),
            buf.c_obj)
        return col
    
    def size(self):
        return self.c_obj[0].size()
    
    def __dealloc__(self):
        del self.c_obj
            
cdef class Column:
    def __cinit__(self):
        pass

    @classmethod
    def from_array(cls, array):
        cdef Column col = Column.__new__(Column)
        cdef type_id dtype = np_to_cudf_types[array.dtype]
        buf = DeviceBuffer(array)
        col.c_obj = new column(
            data_type(dtype),
            len(array),
            buf.c_obj)
        return col
    
    def size(self):
        return self.c_obj[0].size()
    
    def __dealloc__(self):
        del self.c_obj
        
                         
            
