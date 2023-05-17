# Copyright (c) 2023, NVIDIA CORPORATION.

from . cimport libcudf_classes


cdef class Table:
    """A set of columns of the same size."""
    def __init__(self, object columns):
        self.column = columns

        self._underlying = None

    cpdef libcudf_classes.TableView get_underlying(self):
        if self._underlying is None:
            self._underlying = libcudf_classes.TableView(
                [col.get_underlying() for col in self.columns]
            )
        return self._underlying
