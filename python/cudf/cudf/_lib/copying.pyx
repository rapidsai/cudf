# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pickle

import pandas as pd

from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr, shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libc.stdint cimport int32_t, int64_t, uint8_t, uintptr_t

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.column cimport Column
from cudf._lib.scalar import as_device_scalar
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.table cimport Table
from cudf._lib.reduce import minmax

from cudf.core.abc import Serializable

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)
from cudf._lib.cpp.libcpp.functional cimport reference_wrapper
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.lists.gather cimport (
    segmented_gather as cpp_segmented_gather
)
cimport cudf._lib.cpp.copying as cpp_copying

# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


def copy_column(Column input_column):
    """
    Deep copies a column

    Parameters
    ----------
    input_columns : column to be copied

    Returns
    -------
    Deep copied column
    """

    cdef column_view input_column_view = input_column.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(make_unique[column](input_column_view))

    return Column.from_unique_ptr(move(c_result))


def _copy_range_in_place(Column input_column,
                         Column target_column,
                         size_type input_begin,
                         size_type input_end,
                         size_type target_begin):

    cdef column_view input_column_view = input_column.view()
    cdef mutable_column_view target_column_view = target_column.mutable_view()
    cdef size_type c_input_begin = input_begin
    cdef size_type c_input_end = input_end
    cdef size_type c_target_begin = target_begin

    with nogil:
        cpp_copying.copy_range_in_place(
            input_column_view,
            target_column_view,
            c_input_begin,
            c_input_end,
            c_target_begin)


def _copy_range(Column input_column,
                Column target_column,
                size_type input_begin,
                size_type input_end,
                size_type target_begin):

    cdef column_view input_column_view = input_column.view()
    cdef column_view target_column_view = target_column.view()
    cdef size_type c_input_begin = input_begin
    cdef size_type c_input_end = input_end
    cdef size_type c_target_begin = target_begin

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_copying.copy_range(
            input_column_view,
            target_column_view,
            c_input_begin,
            c_input_end,
            c_target_begin)
        )

    return Column.from_unique_ptr(move(c_result))


def copy_range(Column input_column,
               Column target_column,
               size_type input_begin,
               size_type input_end,
               size_type target_begin,
               size_type target_end,
               bool inplace):
    """
    Copy input_column from input_begin to input_end to
    target_column from target_begin to target_end
    """

    if abs(target_end - target_begin) <= 1:
        return target_column

    if target_begin < 0:
        target_begin = target_begin + target_column.size

    if target_end < 0:
        target_end = target_end + target_column.size

    if target_begin > target_end:
        return target_column

    if inplace is True:
        _copy_range_in_place(input_column, target_column,
                             input_begin, input_end, target_begin)
    else:
        return _copy_range(input_column, target_column,
                           input_begin, input_end, target_begin)


def gather(
    Table source_table,
    Column gather_map,
    bool keep_index=True,
    bool nullify=False
):
    if not pd.api.types.is_integer_dtype(gather_map.dtype):
        raise ValueError("Gather map is not integer dtype.")

    if len(gather_map) > 0 and not nullify:
        gm_min, gm_max = minmax(gather_map)
        if gm_min < -len(source_table) or gm_max >= len(source_table):
            raise IndexError(f"Gather map index with min {gm_min},"
                             f" max {gm_max} is out of bounds in"
                             f" {type(source_table)} with {len(source_table)}"
                             f" rows.")

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view
    if keep_index is True:
        source_table_view = source_table.view()
    else:
        source_table_view = source_table.data_view()
    cdef column_view gather_map_view = gather_map.view()
    cdef cpp_copying.out_of_bounds_policy policy = (
        cpp_copying.out_of_bounds_policy.NULLIFY if nullify
        else cpp_copying.out_of_bounds_policy.DONT_CHECK
    )

    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table_view,
                gather_map_view,
                policy
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if (
                source_table._index is None)
            or keep_index is False
            else source_table._index_names
        )
    )


def _scatter_table(Table source_table, Column scatter_map,
                   Table target_table, bool bounds_check=True):

    cdef table_view source_table_view = source_table.data_view()
    cdef column_view scatter_map_view = scatter_map.view()
    cdef table_view target_table_view = target_table.data_view()
    cdef bool c_bounds_check = bounds_check

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                source_table_view,
                scatter_map_view,
                target_table_view,
                c_bounds_check
            )
        )

    out_table = Table.from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=None
    )

    out_table._index = (
        None if target_table._index is None else target_table._index.copy(
            deep=False)
    )

    return out_table


def _scatter_scalar(scalars, Column scatter_map,
                    Table target_table, bool bounds_check=True):

    cdef vector[reference_wrapper[constscalar]] source_scalars
    source_scalars.reserve(len(scalars))
    cdef bool c_bounds_check = bounds_check
    cdef DeviceScalar slr
    for val, col in zip(scalars, target_table._columns):
        slr = as_device_scalar(val, col.dtype)
        source_scalars.push_back(reference_wrapper[constscalar](
            slr.get_raw_ptr()[0]))
    cdef column_view scatter_map_view = scatter_map.view()
    cdef table_view target_table_view = target_table.data_view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                source_scalars,
                scatter_map_view,
                target_table_view,
                c_bounds_check
            )
        )

    out_table = Table.from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=None
    )

    out_table._index = (
        None if target_table._index is None else target_table._index.copy(
            deep=False)
    )

    return out_table


def scatter(object input, object scatter_map, Table target,
            bool bounds_check=True):
    """
    Scattering input into target as per the scatter map,
    input can be a list of scalars or can be a table
    """

    from cudf.core.column.column import as_column

    if not isinstance(scatter_map, Column):
        scatter_map = as_column(scatter_map)

    if isinstance(input, Table):
        return _scatter_table(input, scatter_map, target, bounds_check)
    else:
        return _scatter_scalar(input, scatter_map, target, bounds_check)


def _reverse_column(Column source_column):
    cdef column_view reverse_column_view = source_column.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_copying.reverse(
            reverse_column_view
        ))

    return Column.from_unique_ptr(
        move(c_result)
    )


def _reverse_table(Table source_table):
    cdef table_view reverse_table_view = source_table.view()

    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(cpp_copying.reverse(
            reverse_table_view
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index_names
    )


def reverse(object source):
    """
    Reversing a column or a table
    """
    if isinstance(source, Column):
        return _reverse_column(source)
    else:
        return _reverse_table(source)


def column_empty_like(Column input_column):

    cdef column_view input_column_view = input_column.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_copying.empty_like(input_column_view))

    return Column.from_unique_ptr(move(c_result))


def column_allocate_like(Column input_column, size=None):

    cdef size_type c_size = 0
    cdef column_view input_column_view = input_column.view()
    cdef unique_ptr[column] c_result

    if size is None:
        with nogil:
            c_result = move(cpp_copying.allocate_like(
                input_column_view,
                cpp_copying.mask_allocation_policy.RETAIN)
            )
    else:
        c_size = size
        with nogil:
            c_result = move(cpp_copying.allocate_like(
                input_column_view,
                c_size,
                cpp_copying.mask_allocation_policy.RETAIN)
            )

    return Column.from_unique_ptr(move(c_result))


def table_empty_like(Table input_table, bool keep_index=True):

    cdef table_view input_table_view
    if keep_index is True:
        input_table_view = input_table.view()
    else:
        input_table_view = input_table.data_view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_copying.empty_like(input_table_view))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=input_table._column_names,
        index_names=(
            input_table._index._column_names if keep_index is True else None
        )
    )


def column_slice(Column input_column, object indices):

    cdef column_view input_column_view = input_column.view()
    cdef vector[size_type] c_indices
    c_indices.reserve(len(indices))

    cdef vector[column_view] c_result

    cdef int index

    for index in indices:
        c_indices.push_back(index)

    with nogil:
        c_result = move(
            cpp_copying.slice(
                input_column_view,
                c_indices)
        )

    num_of_result_cols = c_result.size()
    result = [
        Column.from_column_view(
            c_result[i],
            input_column) for i in range(num_of_result_cols)]

    return result


def table_slice(Table input_table, object indices, bool keep_index=True):

    cdef table_view input_table_view
    if keep_index is True:
        input_table_view = input_table.view()
    else:
        input_table_view = input_table.data_view()

    cdef vector[size_type] c_indices
    c_indices.reserve(len(indices))

    cdef vector[table_view] c_result

    cdef int index
    for index in indices:
        c_indices.push_back(index)

    with nogil:
        c_result = move(
            cpp_copying.slice(
                input_table_view,
                c_indices)
        )

    num_of_result_cols = c_result.size()
    result =[
        Table.from_table_view(
            c_result[i],
            input_table,
            column_names=input_table._column_names,
            index_names=(
                input_table._index._column_names if (
                    keep_index is True)
                else None
            )
        ) for i in range(num_of_result_cols)]

    return result


def column_split(Column input_column, object splits):

    cdef column_view input_column_view = input_column.view()
    cdef vector[size_type] c_splits
    c_splits.reserve(len(splits))

    cdef vector[column_view] c_result

    cdef int split

    for split in splits:
        c_splits.push_back(split)

    with nogil:
        c_result = move(
            cpp_copying.split(
                input_column_view,
                c_splits)
        )

    num_of_result_cols = c_result.size()
    result = [
        Column.from_column_view(
            c_result[i],
            input_column
        ) for i in range(num_of_result_cols)
    ]

    return result


def table_split(Table input_table, object splits, bool keep_index=True):

    cdef table_view input_table_view
    if keep_index is True:
        input_table_view = input_table.view()
    else:
        input_table_view = input_table.data_view()

    cdef vector[size_type] c_splits
    c_splits.reserve(len(splits))

    cdef vector[table_view] c_result

    cdef int split
    for split in splits:
        c_splits.push_back(split)

    with nogil:
        c_result = move(
            cpp_copying.split(
                input_table_view,
                c_splits)
        )

    num_of_result_cols = c_result.size()
    result = [
        Table.from_table_view(
            c_result[i],
            input_table,
            column_names=input_table._column_names,
            index_names=input_table._index_names if (
                keep_index is True)
            else None
        ) for i in range(num_of_result_cols)]

    return result


def _copy_if_else_column_column(Column lhs, Column rhs, Column boolean_mask):

    cdef column_view lhs_view = lhs.view()
    cdef column_view rhs_view = rhs.view()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_copying.copy_if_else(
                lhs_view,
                rhs_view,
                boolean_mask_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _copy_if_else_scalar_column(DeviceScalar lhs,
                                Column rhs,
                                Column boolean_mask):

    cdef const scalar* lhs_scalar = lhs.get_raw_ptr()
    cdef column_view rhs_view = rhs.view()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_copying.copy_if_else(
                lhs_scalar[0],
                rhs_view,
                boolean_mask_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _copy_if_else_column_scalar(Column lhs,
                                DeviceScalar rhs,
                                Column boolean_mask):

    cdef column_view lhs_view = lhs.view()
    cdef const scalar* rhs_scalar = rhs.get_raw_ptr()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_copying.copy_if_else(
                lhs_view,
                rhs_scalar[0],
                boolean_mask_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _copy_if_else_scalar_scalar(DeviceScalar lhs,
                                DeviceScalar rhs,
                                Column boolean_mask):

    cdef const scalar* lhs_scalar = lhs.get_raw_ptr()
    cdef const scalar* rhs_scalar = rhs.get_raw_ptr()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_copying.copy_if_else(
                lhs_scalar[0],
                rhs_scalar[0],
                boolean_mask_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


def copy_if_else(object lhs, object rhs, Column boolean_mask):

    if isinstance(lhs, Column):
        if isinstance(rhs, Column):
            return _copy_if_else_column_column(lhs, rhs, boolean_mask)
        else:
            return _copy_if_else_column_scalar(
                lhs, as_device_scalar(rhs), boolean_mask)
    else:
        if isinstance(rhs, Column):
            return _copy_if_else_scalar_column(
                as_device_scalar(lhs), rhs, boolean_mask)
        else:
            if lhs is None and rhs is None:
                return lhs

            return _copy_if_else_scalar_scalar(
                as_device_scalar(lhs), as_device_scalar(rhs), boolean_mask)


def _boolean_mask_scatter_table(Table input_table, Table target_table,
                                Column boolean_mask):

    cdef table_view input_table_view = input_table.view()
    cdef table_view target_table_view = target_table.view()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.boolean_mask_scatter(
                input_table_view,
                target_table_view,
                boolean_mask_view
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=target_table._index._column_names
    )


def _boolean_mask_scatter_scalar(list input_scalars, Table target_table,
                                 Column boolean_mask):

    cdef vector[reference_wrapper[constscalar]] input_scalar_vector
    input_scalar_vector.reserve(len(input_scalars))
    cdef DeviceScalar scl
    for scl in input_scalars:
        input_scalar_vector.push_back(reference_wrapper[constscalar](
            scl.get_raw_ptr()[0]))
    cdef table_view target_table_view = target_table.view()
    cdef column_view boolean_mask_view = boolean_mask.view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.boolean_mask_scatter(
                input_scalar_vector,
                target_table_view,
                boolean_mask_view
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=target_table._index._column_names
    )


def boolean_mask_scatter(object input, Table target_table,
                         Column boolean_mask):

    if isinstance(input, Table):
        return _boolean_mask_scatter_table(
            input,
            target_table,
            boolean_mask
        )
    else:
        scalar_list = [as_device_scalar(i) for i in input]
        return _boolean_mask_scatter_scalar(
            scalar_list,
            target_table,
            boolean_mask
        )


def shift(Column input, int offset, object fill_value=None):

    cdef DeviceScalar fill

    if isinstance(fill_value, DeviceScalar):
        fill = fill_value
    else:
        fill = as_device_scalar(fill_value, input.dtype)

    cdef column_view c_input = input.view()
    cdef int32_t c_offset = offset
    cdef const scalar* c_fill_value = fill.get_raw_ptr()
    cdef unique_ptr[column] c_output

    with nogil:
        c_output = move(
            cpp_copying.shift(
                c_input,
                c_offset,
                c_fill_value[0]
            )
        )

    return Column.from_unique_ptr(move(c_output))


def get_element(Column input_column, size_type index):
    cdef column_view col_view = input_column.view()

    cdef unique_ptr[scalar] c_output
    with nogil:
        c_output = move(
            cpp_copying.get_element(col_view, index)
        )

    return DeviceScalar.from_unique_ptr(
        move(c_output), dtype=input_column.dtype
    )


def sample(Table input, size_type n,
           bool replace, int64_t seed, bool keep_index=True):
    cdef table_view tbl_view = (
        input.view() if keep_index else input.data_view()
    )
    cdef cpp_copying.sample_with_replacement replacement

    if replace:
        replacement = cpp_copying.sample_with_replacement.TRUE
    else:
        replacement = cpp_copying.sample_with_replacement.FALSE

    cdef unique_ptr[table] c_output
    with nogil:
        c_output = move(
            cpp_copying.sample(tbl_view, n, replacement, seed)
        )

    return Table.from_unique_ptr(
        move(c_output),
        column_names=input._column_names,
        index_names=(
            None if keep_index is False
            else input._index_names
        )
    )


def segmented_gather(Column source_column, Column gather_map):
    cdef shared_ptr[lists_column_view] source_LCV = (
        make_shared[lists_column_view](source_column.view())
    )
    cdef shared_ptr[lists_column_view] gather_map_LCV = (
        make_shared[lists_column_view](gather_map.view())
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_segmented_gather(
                source_LCV.get()[0], gather_map_LCV.get()[0])
        )

    result = Column.from_unique_ptr(move(c_result))
    return result


cdef class _CPackedColumns:

    @staticmethod
    cdef _CPackedColumns from_py_table(Table input_table, keep_index=True):
        """
        Construct a ``PackedColumns`` object from a ``cudf.DataFrame``.
        """
        from cudf.core import RangeIndex, dtypes

        cdef _CPackedColumns p = _CPackedColumns.__new__(_CPackedColumns)

        if keep_index and not input_table.index.equals(
            RangeIndex(start=0, stop=len(input_table), step=1)
        ):
            input_table_view = input_table.view()
            p.index_names = input_table._index_names
        else:
            input_table_view = input_table.data_view()

        p.column_names = input_table._column_names
        p.column_dtypes = {}
        for name, col in input_table._data.items():
            if isinstance(col.dtype, dtypes._BaseDtype):
                p.column_dtypes[name] = col.dtype

        p.c_obj = move(cpp_copying.pack(input_table_view))

        return p

    @property
    def gpu_data_ptr(self):
        return int(<uintptr_t>self.c_obj.gpu_data.get()[0].data())

    @property
    def gpu_data_size(self):
        return int(<size_t>self.c_obj.gpu_data.get()[0].size())

    def serialize(self):
        header = {}
        frames = []

        header["column-names"] = self.column_names
        header["index-names"] = self.index_names
        header["gpu-data-ptr"] = self.gpu_data_ptr
        header["gpu-data-size"] = self.gpu_data_size
        header["metadata"] = list(
            <uint8_t[:self.c_obj.metadata_.get()[0].size()]>
            self.c_obj.metadata_.get()[0].data()
        )

        column_dtypes = {}
        for name, dtype in self.column_dtypes.items():
            dtype_header, dtype_frames = dtype.serialize()
            column_dtypes[name] = (
                dtype_header,
                (len(frames), len(frames) + len(dtype_frames)),
            )
            frames.extend(dtype_frames)
        header["column-dtypes"] = column_dtypes

        return header, frames

    @staticmethod
    def deserialize(header, frames):
        cdef _CPackedColumns p = _CPackedColumns.__new__(_CPackedColumns)

        dbuf = DeviceBuffer(
            ptr=header["gpu-data-ptr"],
            size=header["gpu-data-size"]
        )

        cdef cpp_copying.packed_columns data
        data.metadata_ = move(
            make_unique[cpp_copying.metadata](
                move(<vector[uint8_t]>header["metadata"])
            )
        )
        data.gpu_data = move(dbuf.c_obj)

        p.c_obj = move(data)
        p.column_names = header["column-names"]
        p.index_names = header["index-names"]

        column_dtypes = {}
        for name, dtype in header["column-dtypes"].items():
            dtype_header, (start, stop) = dtype
            column_dtypes[name] = pickle.loads(
                dtype_header["type-serialized"]
            ).deserialize(dtype_header, frames[start:stop])
        p.column_dtypes = column_dtypes

        return p

    def unpack(self):
        output_table = Table.from_table_view(
            cpp_copying.unpack(self.c_obj),
            self,
            self.column_names,
            self.index_names
        )

        for name, dtype in self.column_dtypes.items():
            output_table._data[name] = (
                output_table._data[name]._with_type_metadata(dtype)
            )

        return output_table


class PackedColumns(Serializable):
    """
    A packed representation of a ``cudf.Table``, with all columns residing
    in a single GPU memory buffer.
    """

    def __init__(self, data):
        self._data = data

    def __reduce__(self):
        return self.deserialize, self.serialize()

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self._data.gpu_data_ptr, False),
            "shape": (self._data.gpu_data_size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0
        }

    def serialize(self):
        return self._data.serialize()

    @classmethod
    def deserialize(cls, header, frames):
        return cls(_CPackedColumns.deserialize(header, frames))

    @classmethod
    def from_py_table(cls, input_table, keep_index=True):
        return cls(_CPackedColumns.from_py_table(input_table, keep_index))

    def unpack(self):
        return self._data.unpack()


def pack(input_table, keep_index=True):
    """
    Pack the columns of a ``cudf.Table`` into a single GPU memory buffer.
    """
    return PackedColumns.from_py_table(input_table, keep_index)


def unpack(packed):
    """
    Unpack the results of packing a ``cudf.Table``, returning a new
    ``Table`` in the process.
    """
    return packed.unpack()
