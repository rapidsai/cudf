# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.arrow._cuda cimport *
from pyarrow.includes.libarrow cimport *
from cudf._lib.arrow.libarrow_cuda cimport *

cdef extern from "cudf/ipc.hpp" nogil:

    cdef cppclass CCudaMessageReader" CudaMessageReader"(CMessageReader):
        @staticmethod
        unique_ptr[CMessageReader] Open(CCudaBufferReader* stream,
                                        CBufferReader* schema)
        CStatus ReadNextMessage(unique_ptr[CMessage]* out)

    # This declaration is here because there's currently a bug in the pyarrow
    # Cython type signatures. They're declaring
    # `shared_ptr[CRecordBatchStreamReader]* out`, but the actual type in
    # Arrow C++ is `shared_ptr[CRecordBatchReader]* out`, like we have here
    cdef cppclass CCudaRecordBatchStreamReader \
            " CudaRecordBatchStreamReader"(CRecordBatchStreamReader):
        @staticmethod
        CStatus Open(unique_ptr[CMessageReader] message_reader,
                     shared_ptr[CRecordBatchReader]* out)
