# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import functools
import operator

__all__ = ["gpumemoryview"]

cdef class gpumemoryview:
    """Minimal representation of a memory buffer.

    This class aspires to be a GPU equivalent of :py:class:`memoryview` for any
    objects exposing a `CUDA Array Interface
    <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__.
    It will be expanded to encompass more memoryview functionality over time.
    """
    # TODO: dlpack support
    def __init__(self, object obj):
        try:
            cai = obj.__cuda_array_interface__
        except AttributeError:
            raise ValueError(
                "gpumemoryview must be constructed from an object supporting "
                "the CUDA array interface"
            )
        self.obj = obj
        # TODO: Need to respect readonly
        self.ptr = cai["data"][0]

    @property
    def __cuda_array_interface__(self):
        return self.obj.__cuda_array_interface__

    def __len__(self):
        return self.obj.__cuda_array_interface__["shape"][0]

    @property
    def nbytes(self):
        cai = self.obj.__cuda_array_interface__
        shape, typestr = cai["shape"], cai["typestr"]

        # Get element size from typestr, format is two character specifying
        # the type and the latter part is the number of bytes. E.g., '<f4' for
        # 32-bit (4-byte) float.
        element_size = int(typestr[2:])

        return functools.reduce(operator.mul, shape) * element_size

    __hash__ = None
