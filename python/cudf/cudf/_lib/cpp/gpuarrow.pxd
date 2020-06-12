# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pyarrow.includes.libarrow_cuda cimport CCudaBufferReader
from pyarrow.includes.libarrow cimport (
    CStatus,
    CMessage,
    CBufferReader,
    CMessageReader
)

cdef extern from "cudf/ipc.hpp" nogil:

    cdef cppclass CCudaMessageReader" CudaMessageReader"(CMessageReader):
        @staticmethod
        unique_ptr[CMessageReader] Open(CCudaBufferReader* stream,
                                        CBufferReader* schema)
        CStatus ReadNextMessage(unique_ptr[CMessage]* out)
