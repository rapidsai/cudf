# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pickle

import pandas as pd

from libc.stdint cimport int32_t, int64_t, uint8_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

import cudf
from cudf.core.buffer import Buffer

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport table_view_from_columns, table_view_from_table

from cudf._lib.reduce import minmax
from cudf.core.abc import Serializable

cimport cudf._lib.cpp.copying as cpp_copying
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from cudf._lib.cpp.libcpp.functional cimport reference_wrapper
from cudf._lib.cpp.lists.gather cimport (
    segmented_gather as cpp_segmented_gather,
)
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport (
    columns_from_unique_ptr,
    data_from_table_view,
    data_from_unique_ptr,
    table_view_from_columns,
)

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
    columns: list,
    Column gather_map,
    bool nullify=False
):
    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = table_view_from_columns(columns)
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

    return columns_from_unique_ptr(move(c_result))


def scatter(object source, Column scatter_map, Column target_column,
            bool bounds_check=True):
    """
    Scattering input into target as per the scatter map,
    input can be a list of scalars or can be a table
    """

    cdef column_view scatter_map_view = scatter_map.view()
    cdef table_view target_table_view = table_view_from_columns(
        (target_column,))
    cdef bool c_bounds_check = bounds_check
    cdef unique_ptr[table] c_result

    # Needed for the table branch
    cdef table_view source_table_view

    # Needed for the scalar branch
    cdef vector[reference_wrapper[constscalar]] source_scalars
    cdef DeviceScalar slr

    if isinstance(source, Column):
        source_table_view = table_view_from_columns((<Column> source,))

        with nogil:
            c_result = move(
                cpp_copying.scatter(
                    source_table_view,
                    scatter_map_view,
                    target_table_view,
                    c_bounds_check
                )
            )
    else:
        slr = as_device_scalar(source, target_column.dtype)
        source_scalars.push_back(reference_wrapper[constscalar](
            slr.get_raw_ptr()[0]))

        with nogil:
            c_result = move(
                cpp_copying.scatter(
                    source_scalars,
                    scatter_map_view,
                    target_table_view,
                    c_bounds_check
                )
            )

    data, _ = data_from_unique_ptr(
        move(c_result),
        column_names=(None,),
        index_names=None
    )

    return next(iter(data.values()))


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


def table_empty_like(input_table, bool keep_index=True):

    cdef table_view input_table_view = table_view_from_table(
        input_table, not keep_index
    )

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_copying.empty_like(input_table_view))

    return data_from_unique_ptr(
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


def table_slice(input_table, object indices, bool keep_index=True):

    cdef table_view input_table_view = table_view_from_table(
        input_table, not keep_index
    )

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
    return [
        data_from_table_view(
            c_result[i],
            input_table,
            column_names=input_table._column_names,
            index_names=(
                input_table._index._column_names if (
                    keep_index is True)
                else None
            )
        ) for i in range(num_of_result_cols)]


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


def table_split(input_table, object splits, bool keep_index=True):

    cdef table_view input_table_view = table_view_from_table(
        input_table, not keep_index
    )

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
    return [
        data_from_table_view(
            c_result[i],
            input_table,
            column_names=input_table._column_names,
            index_names=input_table._index_names if (
                keep_index is True)
            else None
        ) for i in range(num_of_result_cols)]


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


def _boolean_mask_scatter_table(input_table, target_table,
                                Column boolean_mask):

    cdef table_view input_table_view = table_view_from_columns(input_table)
    cdef table_view target_table_view = table_view_from_columns(target_table)
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

    return data_from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=target_table._index._column_names
    )


def _boolean_mask_scatter_scalar(list input_scalars, target_table,
                                 Column boolean_mask):

    cdef vector[reference_wrapper[constscalar]] input_scalar_vector
    input_scalar_vector.reserve(len(input_scalars))
    cdef DeviceScalar scl
    for scl in input_scalars:
        input_scalar_vector.push_back(reference_wrapper[constscalar](
            scl.get_raw_ptr()[0]))
    cdef table_view target_table_view = table_view_from_columns(target_table)
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

    return data_from_unique_ptr(
        move(c_result),
        column_names=target_table._column_names,
        index_names=target_table._index._column_names
    )


# TODO: This function is currently unused but should be used in
# ColumnBase.__setitem__, see https://github.com/rapidsai/cudf/issues/8667.
def boolean_mask_scatter(object input, target_table,
                         Column boolean_mask):

    if isinstance(input, cudf.core.frame.Frame):
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


def sample(input, size_type n,
           bool replace, int64_t seed, bool keep_index=True):
    cdef table_view tbl_view = table_view_from_table(input, not keep_index)
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

    return data_from_unique_ptr(
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
    def from_py_table(input_table, keep_index=True):
        """
        Construct a ``PackedColumns`` object from a ``cudf.DataFrame``.
        """
        import cudf.core.dtypes

        cdef _CPackedColumns p = _CPackedColumns.__new__(_CPackedColumns)

        if keep_index and (
            not isinstance(input_table.index, cudf.RangeIndex)
            or input_table.index.start != 0
            or input_table.index.stop != len(input_table)
            or input_table.index.step != 1
        ):
            input_table_view = table_view_from_table(input_table)
            p.index_names = input_table._index_names
        else:
            input_table_view = table_view_from_table(
                input_table, ignore_index=True)

        p.column_names = input_table._column_names
        p.column_dtypes = {}
        for name, col in input_table._data.items():
            if isinstance(col.dtype, cudf.core.dtypes._BaseDtype):
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

        gpu_data = Buffer(self.gpu_data_ptr, self.gpu_data_size, self)
        data_header, data_frames = gpu_data.serialize()
        header["data"] = data_header
        frames.extend(data_frames)

        header["column-names"] = self.column_names
        header["index-names"] = self.index_names
        if self.c_obj.metadata_.get()[0].data() != NULL:
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

        gpu_data = Buffer.deserialize(header["data"], frames)

        dbuf = DeviceBuffer(
            ptr=gpu_data.ptr,
            size=gpu_data.nbytes
        )

        cdef cpp_copying.packed_columns data
        data.metadata_ = move(
            make_unique[cpp_copying.metadata](
                move(<vector[uint8_t]>header.get("metadata", []))
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
        output_table = cudf.core.frame.Frame(*data_from_table_view(
            cpp_copying.unpack(self.c_obj),
            self,
            self.column_names,
            self.index_names
        ))

        for name, dtype in self.column_dtypes.items():
            output_table._data[name] = (
                output_table._data[name]._with_type_metadata(dtype)
            )

        return output_table


class PackedColumns(Serializable):
    """
    A packed representation of a Frame, with all columns residing
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
        header, frames = self._data.serialize()
        header["type-serialized"] = pickle.dumps(type(self))

        return header, frames

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
    Pack the columns of a cudf Frame into a single GPU memory buffer.
    """
    return PackedColumns.from_py_table(input_table, keep_index)


def unpack(packed):
    """
    Unpack the results of packing a cudf Frame returning a new
    cudf Frame in the process.
    """
    return packed.unpack()
