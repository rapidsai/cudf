# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow cimport lib as pa

from dataclasses import dataclass, field

from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.scalar.scalar cimport fixed_point_scalar, scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)

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

    This is the Cython representation of :cpp:class:`cudf::column_metadata`.
    """
    name: str = ""
    children_meta: list[ColumnMetadata] = field(default_factory=list)


# These functions are pure Python functions in anticipation of when we no
# longer use pyarrow's Cython and instead just leverage the capsule interface.
def from_arrow(pyarrow_object, *, DataType data_type=None):
    """Create a cudf object from a pyarrow object.

    Parameters
    ----------
    pyarrow_object : Union[pyarrow.Table, pyarrow.Scalar]
        The PyArrow object to convert.

    Returns
    -------
    Union[Table, Scalar]
        The converted object of type corresponding to the input type in cudf.
    """
    # Variables used in the Table block
    cdef shared_ptr[pa.CTable] arrow_table
    cdef unique_ptr[table] c_table_result

    # Variables used in the Scalar block
    cdef shared_ptr[pa.CScalar] arrow_scalar
    cdef unique_ptr[scalar] c_scalar_result
    cdef Scalar scalar_result
    cdef type_id tid

    if isinstance(pyarrow_object, pa.Table):
        if data_type is not None:
            raise ValueError("data_type may not be passed for tables")
        arrow_table = pa.pyarrow_unwrap_table(pyarrow_object)

        with nogil:
            c_table_result = move(cpp_from_arrow(dereference(arrow_table)))

        return Table.from_libcudf(move(c_table_result))
    elif isinstance(pyarrow_object, pa.Scalar):
        arrow_scalar = pa.pyarrow_unwrap_scalar(pyarrow_object)

        with nogil:
            c_scalar_result = move(cpp_from_arrow(dereference(arrow_scalar)))

        scalar_result = Scalar.from_libcudf(move(c_scalar_result))

        if scalar_result.type().id() != type_id.DECIMAL128:
            if data_type is not None:
                raise ValueError(
                    "dtype may not be passed for non-decimal types"
                )
            return scalar_result

        if data_type is None:
            raise ValueError(
                "Decimal scalars must be constructed with a dtype"
            )

        tid = data_type.id()

        if tid == type_id.DECIMAL32:
            scalar_result.c_obj.reset(
                new fixed_point_scalar[decimal32](
                    (
                        <fixed_point_scalar[decimal128]*> scalar_result.c_obj.get()
                    ).value(),
                    scale_type(-pyarrow_object.type.scale),
                    scalar_result.c_obj.get().is_valid()
                )
            )
        elif tid == type_id.DECIMAL64:
            scalar_result.c_obj.reset(
                new fixed_point_scalar[decimal64](
                    (
                        <fixed_point_scalar[decimal128]*> scalar_result.c_obj.get()
                    ).value(),
                    scale_type(-pyarrow_object.type.scale),
                    scalar_result.c_obj.get().is_valid()
                )
            )
        elif tid != type_id.DECIMAL128:
            raise ValueError(
                "Decimal scalars may only be cast to decimals"
            )

        return scalar_result

    raise TypeError("from_arrow only accepts Table and Scalar objects")


def to_arrow(cudf_object, metadata=None):
    """Convert to a PyArrow Table.

    Parameters
    ----------
    metadata : list
        The metadata to attach to the columns of the table.
    """
    # Variables used in the Table block
    cdef shared_ptr[pa.CTable] c_table_result
    cdef vector[column_metadata] c_table_metadata

    # Variables used in the Scalar block
    cdef shared_ptr[pa.CScalar] c_scalar_result
    cdef column_metadata c_scalar_metadata

    if isinstance(cudf_object, Table):
        for meta in metadata:
            c_table_metadata.push_back(_metadata_to_libcudf(meta))
        with nogil:
            c_table_result = move(
                cpp_to_arrow((<Table> cudf_object).view(), c_table_metadata)
            )

        return pa.pyarrow_wrap_table(c_table_result)
    elif isinstance(cudf_object, Scalar):
        c_scalar_metadata = _metadata_to_libcudf(metadata)
        with nogil:
            c_scalar_result = move(
                cpp_to_arrow(
                    dereference((<Scalar> cudf_object).c_obj), c_scalar_metadata
                )
            )

        return pa.pyarrow_wrap_scalar(c_scalar_result)

    raise TypeError("to_arrow only accepts Table and Scalar objects")
