# Copyright (c) 2023, NVIDIA CORPORATION.

from cython cimport no_gc_clear
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from pyarrow cimport lib as pa

from rmm._lib.memory_resource cimport get_current_device_resource

from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.scalar.scalar cimport fixed_point_scalar, scalar
from cudf._lib.cpp.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)

from .interop cimport ColumnMetadata
from .types cimport DataType, type_id


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the Scalar is in a reference cycle. Removing the tp_clear
# function with the no_gc_clear decoration prevents that. See
# https://github.com/rapidsai/rmm/pull/931 for details.
@no_gc_clear
cdef class Scalar:
    """A scalar value in device memory."""
    # Unlike for columns, libcudf does not support scalar views. All APIs that
    # accept scalar values accept references to the owning object rather than a
    # special view type. As a result, pylibcudf.Scalar has a simpler structure
    # than pylibcudf.Column because it can be a true wrapper around a libcudf
    # column

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    def __init__(self, pa.Scalar value=None):
        # TODO: This case is not something we really want to
        # support, but it here for now to ease the transition of
        # DeviceScalar.
        if value is not None:
            raise ValueError("Scalar should be constructed with a factory")

    @staticmethod
    def from_arrow(pa.Scalar value, DataType data_type=None):
        # Allow passing a dtype, but only for the purpose of decimals for now

        cdef shared_ptr[pa.CScalar] cscalar = (
            pa.pyarrow_unwrap_scalar(value)
        )
        cdef unique_ptr[scalar] c_result

        with nogil:
            c_result = move(cpp_from_arrow(cscalar.get()[0]))

        cdef Scalar s = Scalar.from_libcudf(move(c_result))

        if s.type().id() != type_id.DECIMAL128:
            if data_type is not None:
                raise ValueError(
                    "dtype may not be passed for non-decimal types"
                )
            return s

        if data_type is None:
            raise ValueError(
                "Decimal scalars must be constructed with a dtype"
            )

        cdef type_id tid = data_type.id()

        if tid == type_id.DECIMAL32:
            s.c_obj.reset(
                new fixed_point_scalar[decimal32](
                    (<fixed_point_scalar[decimal128]*> s.c_obj.get()).value(),
                    scale_type(-value.type.scale),
                    s.c_obj.get().is_valid()
                )
            )
        elif tid == type_id.DECIMAL64:
            s.c_obj.reset(
                new fixed_point_scalar[decimal64](
                    (<fixed_point_scalar[decimal128]*> s.c_obj.get()).value(),
                    scale_type(-value.type.scale),
                    s.c_obj.get().is_valid()
                )
            )
        elif tid != type_id.DECIMAL128:
            raise ValueError(
                "Decimal scalars may only be cast to decimals"
            )

        return s

    cpdef pa.Scalar to_arrow(self, ColumnMetadata metadata):
        cdef shared_ptr[pa.CScalar] c_result
        cdef column_metadata c_metadata = metadata.to_libcudf()

        with nogil:
            c_result = move(cpp_to_arrow(dereference(self.c_obj.get()), c_metadata))

        return pa.pyarrow_wrap_scalar(c_result)

    cdef const scalar* get(self) noexcept nogil:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self):
        """True if the scalar is valid, false if not"""
        return self.get().is_valid()

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=None):
        """Construct a Scalar object from a libcudf scalar.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(libcudf_scalar)
        s._data_type = DataType.from_libcudf(s.get().type())
        return s
