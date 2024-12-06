# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pickle

from libcpp cimport bool
import pylibcudf

import cudf
from cudf.core.buffer import acquire_spill_lock, as_buffer
from cudf.core.abc import Serializable
from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar

from cudf._lib.reduce import minmax

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.utils cimport columns_from_pylibcudf_table, data_from_pylibcudf_table
import pylibcudf as plc
from pylibcudf.contiguous_split cimport PackedColumns as PlcPackedColumns


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
    return Column.from_pylibcudf(
        input_column.to_pylibcudf(mode="read").copy()
    )


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


class PackedColumns(Serializable):
    """
    A packed representation of a Frame, with all columns residing
    in a single GPU memory buffer.
    """

    def __init__(
        self,
        PlcPackedColumns data,
        object column_names = None,
        object index_names = None,
        object column_dtypes = None
    ):
        self._metadata, self._gpu_data = data.release()
        self.column_names=column_names
        self.index_names=index_names
        self.column_dtypes=column_dtypes

    def __reduce__(self):
        return self.deserialize, self.serialize()

    @property
    def __cuda_array_interface__(self):
        return self._gpu_data.__cuda_array_interface__

    def serialize(self):
        header = {}
        frames = []
        gpu_data = as_buffer(
            data = self._gpu_data.obj.ptr,
            size = self._gpu_data.obj.size,
            owner=self,
            exposed=True
        )
        data_header, data_frames = gpu_data.serialize()
        header["data"] = data_header
        frames.extend(data_frames)

        header["column-names"] = self.column_names
        header["index-names"] = self.index_names
        header["metadata"] = self._metadata.tobytes()
        for name, dtype in self.column_dtypes.items():
            dtype_header, dtype_frames = dtype.serialize()
            self.column_dtypes[name] = (
                dtype_header,
                (len(frames), len(frames) + len(dtype_frames)),
            )
            frames.extend(dtype_frames)
        header["column-dtypes"] = self.column_dtypes
        header["type-serialized"] = pickle.dumps(type(self))
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        column_dtypes = {}
        for name, dtype in header["column-dtypes"].items():
            dtype_header, (start, stop) = dtype
            column_dtypes[name] = pickle.loads(
                dtype_header["type-serialized"]
            ).deserialize(dtype_header, frames[start:stop])
        return cls(
            plc.contiguous_split.pack(
                plc.contiguous_split.unpack_from_memoryviews(
                    memoryview(header["metadata"]),
                    plc.gpumemoryview(frames[0]),
                )
            ),
            header["column-names"],
            header["index-names"],
            column_dtypes,
        )

    @classmethod
    def from_py_table(cls, input_table, keep_index=True):
        if keep_index and (
            not isinstance(input_table.index, cudf.RangeIndex)
            or input_table.index.start != 0
            or input_table.index.stop != len(input_table)
            or input_table.index.step != 1
        ):
            columns = input_table._index._columns + input_table._columns
            index_names = input_table._index_names
        else:
            columns = input_table._columns
            index_names = None

        column_names = input_table._column_names
        column_dtypes = {}
        for name, col in input_table._column_labels_and_values:
            if isinstance(
                col.dtype,
                (cudf.core.dtypes._BaseDtype, cudf.core.dtypes.CategoricalDtype)
            ):
                column_dtypes[name] = col.dtype

        return cls(
            plc.contiguous_split.pack(
                plc.Table(
                    [
                        col.to_pylibcudf(mode="read") for col in columns
                    ]
                )
            ),
            column_names,
            index_names,
            column_dtypes,
        )

    def unpack(self):
        output_table = cudf.DataFrame._from_data(*data_from_pylibcudf_table(
            plc.contiguous_split.unpack_from_memoryviews(
                self._metadata,
                self._gpu_data
            ),
            self.column_names,
            self.index_names
        ))
        for name, dtype in self.column_dtypes.items():
            output_table._data[name] = (
                output_table._data[name]._with_type_metadata(dtype)
            )

        return output_table


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
