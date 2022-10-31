# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import pickle
import warnings

from libc.stdint cimport int32_t, uint8_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

import cudf
from cudf.core.buffer import Buffer, as_buffer

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
    columns_from_table_view,
    columns_from_unique_ptr,
    data_from_table_view,
    table_view_from_columns,
)

# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


def _gather_map_is_valid(
    gather_map: "cudf.core.column.ColumnBase",
    nrows: int,
    check_bounds: bool,
    nullify: bool,
) -> bool:
    """Returns true if gather map is valid.

    A gather map is valid if empty or all indices are within the range
    ``[-nrows, nrows)``, except when ``nullify`` is specifed.
    """
    if not check_bounds or nullify or len(gather_map) == 0:
        return True
    gm_min, gm_max = minmax(gather_map)
    return gm_min >= -nrows and gm_max < nrows


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
    list columns,
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


cdef scatter_scalar(list source_device_slrs,
                    column_view scatter_map,
                    table_view target_table):
    cdef vector[reference_wrapper[constscalar]] c_source
    cdef DeviceScalar d_slr
    cdef unique_ptr[table] c_result

    c_source.reserve(len(source_device_slrs))
    for d_slr in source_device_slrs:
        c_source.push_back(
            reference_wrapper[constscalar](d_slr.get_raw_ptr()[0])
        )

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                c_source,
                scatter_map,
                target_table,
            )
        )

    return columns_from_unique_ptr(move(c_result))


cdef scatter_column(list source_columns,
                    column_view scatter_map,
                    table_view target_table):
    cdef table_view c_source = table_view_from_columns(source_columns)
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                c_source,
                scatter_map,
                target_table,
            )
        )
    return columns_from_unique_ptr(move(c_result))


def scatter(list sources, Column scatter_map, list target_columns,
            bool bounds_check=True):
    """
    Scattering source into target as per the scatter map.
    `source` can be a list of scalars, or a list of columns. The number of
    items in `sources` must equal the number of `target_columns` to scatter.
    """
    # TODO: Only single column scatter is used, we should explore multi-column
    # scatter for frames for performance increase.

    if len(sources) != len(target_columns):
        raise ValueError("Mismatched number of source and target columns.")

    if len(sources) == 0:
        return []

    cdef column_view scatter_map_view = scatter_map.view()
    cdef table_view target_table_view = table_view_from_columns(target_columns)

    if bounds_check:
        n_rows = len(target_columns[0])
        if not (
            (scatter_map >= -n_rows).all()
            and (scatter_map < n_rows).all()
        ):
            raise IndexError(
                f"index out of bounds for column of size {n_rows}"
            )

    if isinstance(sources[0], Column):
        return scatter_column(
            sources, scatter_map_view, target_table_view
        )
    else:
        source_scalars = [as_device_scalar(slr) for slr in sources]
        return scatter_scalar(
            source_scalars, scatter_map_view, target_table_view
        )


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


def columns_empty_like(list input_columns):
    cdef table_view input_table_view = table_view_from_columns(input_columns)
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_copying.empty_like(input_table_view))

    return columns_from_unique_ptr(move(c_result))


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


def columns_slice(list input_columns, list indices):
    """
    Given a list of input columns, return columns sliced by ``indices``.

    Returns a list of list of columns. The length of return is
    `len(indices) / 2`. The `i`th item in return is a list of columns sliced
    from ``input_columns`` with `slice(indices[i*2], indices[i*2 + 1])`.
    """
    cdef table_view input_table_view = table_view_from_columns(input_columns)
    cdef vector[size_type] c_indices = indices
    cdef vector[table_view] c_result

    with nogil:
        c_result = move(
            cpp_copying.slice(
                input_table_view,
                c_indices)
        )

    return [
        columns_from_table_view(
            c_result[i], input_columns
        ) for i in range(c_result.size())
    ]


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


def columns_split(list input_columns, object splits):

    cdef table_view input_table_view = table_view_from_columns(input_columns)
    cdef vector[size_type] c_splits = splits
    cdef vector[table_view] c_result

    with nogil:
        c_result = move(
            cpp_copying.split(
                input_table_view,
                c_splits)
        )

    return [
        columns_from_table_view(
            c_result[i], input_columns
        ) for i in range(c_result.size())
    ]


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


def _boolean_mask_scatter_columns(list input_columns, list target_columns,
                                  Column boolean_mask):

    cdef table_view input_table_view = table_view_from_columns(input_columns)
    cdef table_view target_table_view = table_view_from_columns(target_columns)
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

    return columns_from_unique_ptr(move(c_result))


def _boolean_mask_scatter_scalar(list input_scalars, list target_columns,
                                 Column boolean_mask):

    cdef vector[reference_wrapper[constscalar]] input_scalar_vector
    input_scalar_vector.reserve(len(input_scalars))
    cdef DeviceScalar scl
    for scl in input_scalars:
        input_scalar_vector.push_back(reference_wrapper[constscalar](
            scl.get_raw_ptr()[0]))
    cdef table_view target_table_view = table_view_from_columns(target_columns)
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

    return columns_from_unique_ptr(move(c_result))


def boolean_mask_scatter(list input_, list target_columns,
                         Column boolean_mask):
    """Copy the target columns, replacing masked rows with input data.

    The ``input_`` data can be a list of columns or as a list of scalars.
    A list of input columns will be used to replace corresponding rows in the
    target columns for which the boolean mask is ``True``. For the nth ``True``
    in the boolean mask, the nth row in ``input_`` is used to replace. A list
    of input scalars will replace all rows in the target columns for which the
    boolean mask is ``True``.
    """
    if len(input_) != len(target_columns):
        raise ValueError("Mismatched number of input and target columns.")

    if len(input_) == 0:
        return []

    if isinstance(input_[0], Column):
        return _boolean_mask_scatter_columns(
            input_,
            target_columns,
            boolean_mask
        )
    else:
        scalar_list = [as_device_scalar(i) for i in input_]
        return _boolean_mask_scatter_scalar(
            scalar_list,
            target_columns,
            boolean_mask
        )


def shift(Column input, int offset, object fill_value=None):

    cdef DeviceScalar fill

    if isinstance(fill_value, DeviceScalar):
        fill_value_type = fill_value.dtype
        fill = fill_value
    else:
        fill_value_type = type(fill_value)
        fill = as_device_scalar(fill_value, input.dtype)

    if not cudf.utils.dtypes._can_cast(input.dtype, fill_value_type):
        warnings.warn(
            f"Passing {fill_value_type} to shift is deprecated and will "
            f"raise in a future version"
            f", pass a {input.dtype} scalar instead.",
            FutureWarning,
        )

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

        gpu_data = as_buffer(
            data=self.gpu_data_ptr,
            size=self.gpu_data_size,
            owner=self
        )
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
        output_table = cudf.DataFrame._from_data(*data_from_table_view(
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
