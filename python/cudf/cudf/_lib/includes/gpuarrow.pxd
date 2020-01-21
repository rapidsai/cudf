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
