# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.move cimport move
from cudf._lib.arrow._cuda cimport (
    CudaBuffer
)
from cudf._lib.cpp.arrow.libarrow_cuda cimport (
    CCudaBufferReader
)
from cudf._lib.cpp.gpuarrow cimport CCudaMessageReader
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from pyarrow.includes.libarrow cimport (
    CMessage,
    CBufferReader,
    CMessageReader,
    CRecordBatchStreamReader
)
from pyarrow.lib cimport (
    _CRecordBatchReader,
    Buffer,
    Schema,
    check_status,
    pyarrow_wrap_schema
)
import pyarrow as pa


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
            check_status(CRecordBatchStreamReader.Open2(
                move(message_reader), &self.reader
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
