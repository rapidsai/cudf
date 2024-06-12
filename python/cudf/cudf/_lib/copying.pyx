# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pickle

from libc.stdint cimport uint8_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

import cudf
from cudf._lib import pylibcudf
from cudf.core.buffer import Buffer, acquire_spill_lock, as_buffer

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport table_view_from_table

from cudf._lib.reduce import minmax
from cudf.core.abc import Serializable

from libcpp.memory cimport make_unique

cimport cudf._lib.pylibcudf.libcudf.contiguous_split as cpp_contiguous_split
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.lists.gather cimport (
    segmented_gather as cpp_segmented_gather,
)
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.utils cimport columns_from_pylibcudf_table, data_from_table_view

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
    ``[-nrows, nrows)``, except when ``nullify`` is specified.
    """
    if not check_bounds or nullify or len(gather_map) == 0:
        return True
    gm_min, gm_max = minmax(gather_map)
    return gm_min >= -nrows and gm_max < nrows


@acquire_spill_lock()
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
    cdef unique_ptr[column] c_result
    cdef column_view input_column_view = input_column.view()
    with nogil:
        c_result = move(make_unique[column](input_column_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def _copy_range_in_place(Column input_column,
                         Column target_column,
                         size_type input_begin,
                         size_type input_end,
                         size_type target_begin):
    pylibcudf.copying.copy_range(
        input_column.to_pylibcudf(mode="write"),
        target_column.to_pylibcudf(mode="write"),
        input_begin,
        input_end,
        target_begin
    )


def _copy_range(Column input_column,
                Column target_column,
                size_type input_begin,
                size_type input_end,
                size_type target_begin):
    return Column.from_pylibcudf(
        pylibcudf.copying.copy_range(
            input_column.to_pylibcudf(mode="read"),
            target_column.to_pylibcudf(mode="read"),
            input_begin,
            input_end,
            target_begin
        )
    )


@acquire_spill_lock()
def copy_range(Column source_column,
               Column target_column,
               size_type source_begin,
               size_type source_end,
               size_type target_begin,
               size_type target_end,
               bool inplace):
    """
    Copy a contiguous range from a source to a target column

    Notes
    -----
    Expects the source and target ranges to have been sanitised to be
    in-range for the source and target column respectively. For
    example via ``slice.indices``.
    """

    msg = "Source and target ranges must be same length"
    assert source_end - source_begin == target_end - target_begin, msg
    if target_end >= target_begin and inplace:
        # FIXME: Are we allowed to do this when inplace=False?
        return target_column

    if inplace:
        _copy_range_in_place(source_column, target_column,
                             source_begin, source_end, target_begin)
    else:
        return _copy_range(source_column, target_column,
                           source_begin, source_end, target_begin)


@acquire_spill_lock()
def gather(
    list columns,
    Column gather_map,
    bool nullify=False
):
    tbl = pylibcudf.copying.gather(
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in columns]),
        gather_map.to_pylibcudf(mode="read"),
        pylibcudf.copying.OutOfBoundsPolicy.NULLIFY if nullify
        else pylibcudf.copying.OutOfBoundsPolicy.DONT_CHECK
    )
    return columns_from_pylibcudf_table(tbl)


@acquire_spill_lock()
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

    if bounds_check:
        n_rows = len(target_columns[0])
        if not (
            (scatter_map >= -n_rows).all()
            and (scatter_map < n_rows).all()
        ):
            raise IndexError(
                f"index out of bounds for column of size {n_rows}"
            )

    tbl = pylibcudf.copying.scatter(
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in sources])
        if isinstance(sources[0], Column)
        else [(<DeviceScalar> as_device_scalar(slr)).c_value for slr in sources],
        scatter_map.to_pylibcudf(mode="read"),
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in target_columns]),
    )

    return columns_from_pylibcudf_table(tbl)


@acquire_spill_lock()
def column_empty_like(Column input_column):
    return Column.from_pylibcudf(
        pylibcudf.copying.empty_like(
            input_column.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def column_allocate_like(Column input_column, size=None):
    return Column.from_pylibcudf(
        pylibcudf.copying.allocate_like(
            input_column.to_pylibcudf(mode="read"),
            size,
        )
    )


@acquire_spill_lock()
def columns_empty_like(list input_columns):
    return columns_from_pylibcudf_table(
        pylibcudf.copying.empty_like(
            pylibcudf.Table([col.to_pylibcudf(mode="read") for col in input_columns])
        )
    )


@acquire_spill_lock()
def column_slice(Column input_column, object indices):
    return [
        Column.from_pylibcudf(c)
        for c in pylibcudf.copying.slice(
            input_column.to_pylibcudf(mode="read"),
            list(indices),
        )
    ]


@acquire_spill_lock()
def columns_slice(list input_columns, object indices):
    return [
        columns_from_pylibcudf_table(tbl)
        for tbl in pylibcudf.copying.slice(
            pylibcudf.Table([col.to_pylibcudf(mode="read") for col in input_columns]),
            list(indices),
        )
    ]


@acquire_spill_lock()
def column_split(Column input_column, object splits):
    return [
        Column.from_pylibcudf(c)
        for c in pylibcudf.copying.split(
            input_column.to_pylibcudf(mode="read"),
            list(splits),
        )
    ]


@acquire_spill_lock()
def columns_split(list input_columns, object splits):
    return [
        columns_from_pylibcudf_table(tbl)
        for tbl in pylibcudf.copying.split(
            pylibcudf.Table([col.to_pylibcudf(mode="read") for col in input_columns]),
            list(splits),
        )
    ]


@acquire_spill_lock()
def copy_if_else(object lhs, object rhs, Column boolean_mask):
    return Column.from_pylibcudf(
        pylibcudf.copying.copy_if_else(
            lhs.to_pylibcudf(mode="read") if isinstance(lhs, Column)
            else (<DeviceScalar> as_device_scalar(lhs)).c_value,
            rhs.to_pylibcudf(mode="read") if isinstance(rhs, Column)
            else (<DeviceScalar> as_device_scalar(rhs)).c_value,
            boolean_mask.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
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

    tbl = pylibcudf.copying.boolean_mask_scatter(
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in input_])
        if isinstance(input_[0], Column)
        else [(<DeviceScalar> as_device_scalar(i)).c_value for i in input_],
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in target_columns]),
        boolean_mask.to_pylibcudf(mode="read"),
    )

    return columns_from_pylibcudf_table(tbl)


@acquire_spill_lock()
def shift(Column input, int offset, object fill_value=None):
    cdef DeviceScalar fill

    if isinstance(fill_value, DeviceScalar):
        fill = fill_value
    else:
        fill = as_device_scalar(fill_value, input.dtype)

    col = pylibcudf.copying.shift(
        input.to_pylibcudf(mode="read"),
        offset,
        fill.c_value,
    )
    return Column.from_pylibcudf(col)


@acquire_spill_lock()
def get_element(Column input_column, size_type index):
    return DeviceScalar.from_pylibcudf(
        pylibcudf.copying.get_element(
            input_column.to_pylibcudf(mode="read"),
            index,
        ),
        dtype=input_column.dtype,
    )


@acquire_spill_lock()
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

        p.c_obj = move(cpp_contiguous_split.pack(input_table_view))

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
            owner=self,
            exposed=True
        )
        data_header, data_frames = gpu_data.serialize()
        header["data"] = data_header
        frames.extend(data_frames)

        header["column-names"] = self.column_names
        header["index-names"] = self.index_names
        if self.c_obj.metadata.get()[0].data() != NULL:
            header["metadata"] = list(
                <uint8_t[:self.c_obj.metadata.get()[0].size()]>
                self.c_obj.metadata.get()[0].data()
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
            ptr=gpu_data.get_ptr(mode="write"),
            size=gpu_data.nbytes
        )

        cdef cpp_contiguous_split.packed_columns data
        data.metadata = move(
            make_unique[vector[uint8_t]](
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
            cpp_contiguous_split.unpack(self.c_obj),
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
