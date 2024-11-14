# Copyright (c) 2023-2024, NVIDIA CORPORATION.

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

    __hash__ = None
