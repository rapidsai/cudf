# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import pyarrow as pa
from pyarrow.lib cimport *
from cudf._lib.cudf import *
from cudf._lib.cudf cimport *
from cudf._lib.arrow._cuda cimport *
from cudf._lib.arrow.libarrow_cuda cimport *
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from cudf._lib.includes.gpuarrow cimport *


cdef class CudaRecordBatchStreamReader(_CRecordBatchReader):

    cdef readonly:
        Schema schema

    def __cinit__(self):
        pass

    def _open(self, source, schema=None):

        cdef unique_ptr[CMessageReader] message_reader
        cdef CCudaBufferReader* data_ = to_buffer_reader(source)
        cdef CBufferReader* schema_ = schema_to_buffer_reader(schema)

        with nogil:
            message_reader = CCudaMessageReader.Open(data_, schema_)
            check_status(CCudaRecordBatchStreamReader.Open(
                unique_ptr[CMessageReader](message_reader.release()),
                &self.reader
            ))

        self.schema = pyarrow_wrap_schema(self.reader.get().schema())

cdef CBufferReader* schema_to_buffer_reader(schema):
    cdef Buffer host_buf
    if schema is None:
        host_buf = pa.py_buffer(bytearray(0))
    elif isinstance(schema, pa.Schema):
        host_buf = <Buffer> schema.serialize()
    else:
        host_buf = <Buffer> as_pa_buffer(schema)
    return new CBufferReader(host_buf.buffer)

cdef CCudaBufferReader* to_buffer_reader(object obj):
    cdef CudaBuffer cuda_buf
    if pyarrow_is_cudabuffer(obj):
        cuda_buf = <CudaBuffer> obj
    elif isinstance(obj, DeviceNDArray):
        cuda_buf = CudaBuffer.from_numba(obj.gpu_data)
    else:
        raise ValueError('unrecognized device buffer')
    return new CCudaBufferReader(cuda_buf.buffer)

cdef public api bint pyarrow_is_cudabuffer(object buffer):
    return isinstance(buffer, CudaBuffer)


def as_pa_buffer(object o):
    if isinstance(o, pa.Buffer):
        return o
    return pa.py_buffer(o)
