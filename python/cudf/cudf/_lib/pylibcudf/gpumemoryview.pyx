# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    """Minimal representation of a memory buffer.

    This class aspires to be a GPU equivalent of the [Python memoryview
    type](https://docs.python.org/3/library/stdtypes.html#memoryview) for any
    objects exposing a [CUDA Array
    Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html).
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
        self._obj = obj
        self._released = False
        # TODO: Need to respect readonly
        self.ptr = cai["data"][0]

    def __cuda_array_interface__(self):
        return self.obj.__cuda_array_interface__

    @property
    def obj(self):
        if not self._released:
            return self._obj
        else:
            raise ValueError("operation forbidden on released gpumemoryview object")

    cpdef release(self):
        self._obj = None
        self._released = True
