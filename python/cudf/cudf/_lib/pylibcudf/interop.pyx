# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cpython cimport pycapsule
from cython.operator cimport dereference
from libc.stdlib cimport free
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow cimport lib as pa

from dataclasses import dataclass, field
from functools import singledispatch

from pyarrow import lib as pa

# from cuda cimport ccudart

from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.scalar.scalar cimport fixed_point_scalar, scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType, type_id


cdef column_metadata _metadata_to_libcudf(metadata):
    """Convert a ColumnMetadata object to C++ column_metadata.

    Since this class is mutable and cheap, it is easier to create the C++
    object on the fly rather than have it directly backing the storage for
    the Cython class. Additionally, this structure restricts the dependency
    on C++ types to just within this module, allowing us to make the module a
    pure Python module (from an import sense, i.e. no pxd declarations).
    """
    cdef column_metadata c_metadata
    c_metadata.name = metadata.name.encode()
    for child_meta in metadata.children_meta:
        c_metadata.children_meta.push_back(_metadata_to_libcudf(child_meta))
    return c_metadata


@dataclass
class ColumnMetadata:
    """Metadata associated with a column.

    This is the Python representation of :cpp:class:`cudf::column_metadata`.
    """
    name: str = ""
    children_meta: list[ColumnMetadata] = field(default_factory=list)


@singledispatch
def from_arrow(pyarrow_object, *, DataType data_type=None):
    """Create a cudf object from a pyarrow object.

    Parameters
    ----------
    pyarrow_object : Union[pyarrow.Array, pyarrow.Table, pyarrow.Scalar]
        The PyArrow object to convert.

    Returns
    -------
    Union[Table, Scalar]
        The converted object of type corresponding to the input type in cudf.
    """
    raise TypeError("from_arrow only accepts Table and Scalar objects")


@from_arrow.register(pa.Table)
def _from_arrow_table(pyarrow_object, *, DataType data_type=None):
    if data_type is not None:
        raise ValueError("data_type may not be passed for tables")
    cdef shared_ptr[pa.CTable] arrow_table = pa.pyarrow_unwrap_table(pyarrow_object)

    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(cpp_from_arrow(dereference(arrow_table)))

    return Table.from_libcudf(move(c_result))


@from_arrow.register(pa.Scalar)
def _from_arrow_scalar(pyarrow_object, *, DataType data_type=None):
    cdef shared_ptr[pa.CScalar] arrow_scalar = pa.pyarrow_unwrap_scalar(pyarrow_object)

    cdef unique_ptr[scalar] c_result
    with nogil:
        c_result = move(cpp_from_arrow(dereference(arrow_scalar)))

    cdef Scalar result = Scalar.from_libcudf(move(c_result))

    if result.type().id() != type_id.DECIMAL128:
        if data_type is not None:
            raise ValueError(
                "dtype may not be passed for non-decimal types"
            )
        return result

    if data_type is None:
        raise ValueError(
            "Decimal scalars must be constructed with a dtype"
        )

    cdef type_id tid = data_type.id()

    if tid == type_id.DECIMAL32:
        result.c_obj.reset(
            new fixed_point_scalar[decimal32](
                (
                    <fixed_point_scalar[decimal128]*> result.c_obj.get()
                ).value(),
                scale_type(-pyarrow_object.type.scale),
                result.c_obj.get().is_valid()
            )
        )
    elif tid == type_id.DECIMAL64:
        result.c_obj.reset(
            new fixed_point_scalar[decimal64](
                (
                    <fixed_point_scalar[decimal128]*> result.c_obj.get()
                ).value(),
                scale_type(-pyarrow_object.type.scale),
                result.c_obj.get().is_valid()
            )
        )
    elif tid != type_id.DECIMAL128:
        raise ValueError(
            "Decimal scalars may only be cast to decimals"
        )

    return result


@from_arrow.register(pa.Array)
@from_arrow.register(pa.ChunkedArray)
def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
    if data_type is not None:
        raise ValueError("data_type may not be passed for arrays")
    pa_table = pa.table([pyarrow_object], [""])
    return from_arrow(pa_table).columns()[0]


@singledispatch
def to_arrow(cudf_object, metadata=None):
    """Convert to a PyArrow object.

    Parameters
    ----------
    cudf_object : Union[Column, Table, Scalar]
        The cudf object to convert.
    metadata : list
        The metadata to attach to the columns of the table.

    Returns
    -------
    Union[pyarrow.Array, pyarrow.Table, pyarrow.Scalar]
        The converted object of type corresponding to the input type in PyArrow.
    """
    raise TypeError("to_arrow only accepts Table and Scalar objects")


@to_arrow.register(Table)
def _to_arrow_table(cudf_object, metadata=None):
    if metadata is None:
        metadata = [ColumnMetadata() for _ in range(len(cudf_object.columns()))]
    metadata = [ColumnMetadata(m) if isinstance(m, str) else m for m in metadata]
    cdef vector[column_metadata] c_table_metadata
    cdef shared_ptr[pa.CTable] c_table_result
    for meta in metadata:
        c_table_metadata.push_back(_metadata_to_libcudf(meta))
    with nogil:
        c_table_result = move(
            cpp_to_arrow((<Table> cudf_object).view(), c_table_metadata)
        )

    return pa.pyarrow_wrap_table(c_table_result)


@to_arrow.register(Scalar)
def _to_arrow_scalar(cudf_object, metadata=None):
    # Note that metadata for scalars is primarily important for preserving
    # information on nested types since names are otherwise irrelevant.
    if metadata is None:
        metadata = ColumnMetadata()
    metadata = ColumnMetadata(metadata) if isinstance(metadata, str) else metadata
    cdef column_metadata c_scalar_metadata = _metadata_to_libcudf(metadata)
    cdef shared_ptr[pa.CScalar] c_scalar_result
    with nogil:
        c_scalar_result = move(
            cpp_to_arrow(
                dereference((<Scalar> cudf_object).c_obj), c_scalar_metadata
            )
        )

    return pa.pyarrow_wrap_scalar(c_scalar_result)


@to_arrow.register(Column)
def _to_arrow_array(cudf_object, metadata=None):
    """Create a PyArrow array from a pylibcudf column."""
    if metadata is None:
        metadata = ColumnMetadata()
    metadata = ColumnMetadata(metadata) if isinstance(metadata, str) else metadata
    return to_arrow(Table([cudf_object]), [metadata])[0]


cdef void release_arrow_schema_py_capsule(object schema_capsule) noexcept:
    cdef ArrowSchema* schema = <ArrowSchema*>pycapsule.PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    if schema.release != NULL:
        schema.release(schema)

    free(schema)


cdef extern from *:
    # Rather than exporting the underlying functions directly to Cython, we expose
    # these wrappers that handle the release to avoid needing to teach Cython how
    # to handle unique_ptrs with custom deleters that aren't default constructible.
    """
    ArrowSchema* to_arrow_schema_raw(
      cudf::table_view const& input,
      cudf::host_span<cudf::column_metadata const> metadata) {
      return to_arrow_schema(input, metadata).release();
    }
    ArrowDeviceArray* to_arrow_device_raw(
      cudf::table_view tbl,
      rmm::cuda_stream_view stream        = cudf::get_default_stream(),
      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
      return cudf::to_arrow_device(tbl, stream, mr).release();
    }
    """
    cdef ArrowSchema *to_arrow_schema_raw(
        const table_view& tbl,
        const vector[column_metadata]& metadata,
    ) except + nogil
    cdef ArrowDeviceArray* to_arrow_device_raw(table_view tbl) except + nogil

    # The minimal declarations for Arrow C Data Interface types needed for the
    # Cython implementations.
    cdef struct ArrowSchema:
        void (*release)(ArrowSchema*) noexcept nogil

    cdef struct ArrowArray:
        void (*release)(ArrowArray*) noexcept nogil

    cdef struct ArrowArrayStream:
        void (*release)(ArrowArrayStream*) noexcept nogil

    cdef struct ArrowDeviceArray:
        ArrowArray array


def table_to_schema(Table tbl, metadata):
    if metadata is None:
        metadata = [ColumnMetadata() for _ in range(len(tbl.columns()))]
    metadata = [ColumnMetadata(m) if isinstance(m, str) else m for m in metadata]

    cdef vector[column_metadata] c_metadata
    for meta in metadata:
        c_metadata.push_back(_metadata_to_libcudf(meta))

    cdef ArrowSchema* raw_schema_ptr
    with nogil:
        raw_schema_ptr = to_arrow_schema_raw(tbl.view(), c_metadata)

    capsule = pycapsule.PyCapsule_New(
        <void*>raw_schema_ptr,
        'arrow_schema',
        release_arrow_schema_py_capsule,
    )
    return capsule


cdef void release_arrow_device_array_py_capsule(object device_array_capsule) noexcept:
    cdef ArrowDeviceArray* device_array = (
        <ArrowDeviceArray*>pycapsule.PyCapsule_GetPointer(
            device_array_capsule, 'arrow_device_array'
        )
    )
    if device_array.array.release != NULL:
        device_array.array.release(&device_array.array)

    free(device_array)


def table_to_device_array(Table tbl):
    cdef ArrowDeviceArray* raw_device_array_ptr
    with nogil:
        raw_device_array_ptr = to_arrow_device_raw(tbl.view())

    capsule = pycapsule.PyCapsule_New(
        <void*>raw_device_array_ptr,
        'arrow_device_array',
        release_arrow_device_array_py_capsule,
    )
    return capsule
