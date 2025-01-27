# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from __future__ import annotations

import enum
from collections import abc
from typing import TYPE_CHECKING, Any, cast

import cupy as cp
import numpy as np
from numba.cuda import as_cuda_array

import rmm

import cudf
from cudf.core.buffer import Buffer, as_buffer
from cudf.core.column import (
    CategoricalColumn,
    NumericalColumn,
    as_column,
    build_column,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Implementation of interchange protocol classes
# ----------------------------------------------


class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


class _Device(enum.IntEnum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class _MaskKind(enum.IntEnum):
    NON_NULLABLE = 0
    NAN = 1
    SENTINEL = 2
    BITMASK = 3
    BYTEMASK = 4


_SUPPORTED_KINDS = {
    _DtypeKind.INT,
    _DtypeKind.UINT,
    _DtypeKind.FLOAT,
    _DtypeKind.CATEGORICAL,
    _DtypeKind.BOOL,
    _DtypeKind.STRING,
}
ProtoDtype = tuple[_DtypeKind, int, str, str]


class _CuDFBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(
        self,
        buf: Buffer,
        dtype: np.dtype,
        allow_copy: bool = True,
    ) -> None:
        """
        Use Buffer object.
        """
        # Store the cudf buffer where the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._buf = buf
        self._dtype = dtype
        self._allow_copy = allow_copy

    @property
    def bufsize(self) -> int:
        """
        The Buffer size in bytes.
        """
        return self._buf.size

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._buf.get_ptr(mode="write")

    def __dlpack__(self):
        # DLPack not implemented in NumPy yet, so leave it out here.
        try:
            cuda_array = as_cuda_array(self._buf).view(self._dtype)
            return cp.asarray(cuda_array).toDlpack()
        except ValueError:
            raise TypeError(f"dtype {self._dtype} unsupported by `dlpack`")

    def __dlpack_device__(self) -> tuple[_Device, int]:
        """
        _Device type and _Device ID for where the data in the buffer resides.
        """
        return (_Device.CUDA, cp.asarray(self._buf).device.id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + str(
            {
                "bufsize": self.bufsize,
                "ptr": self.ptr,
                "device": self.__dlpack_device__()[0].name,
            }
        )
        +")"


class _CuDFColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.

    """

    def __init__(
        self,
        column: cudf.core.column.ColumnBase,
        nan_as_null: bool = True,
        allow_copy: bool = True,
    ) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if not isinstance(column, cudf.core.column.ColumnBase):
            raise TypeError(
                "column must be a subtype of df.core.column.ColumnBase,"
                f"got {type(column)}"
            )
        self._col = column
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def size(self) -> int:
        """
        Size of the column, in elements.
        """
        return self._col.size

    @property
    def offset(self) -> int:
        """
        Offset of first element. Always zero.
        """
        return 0

    @property
    def dtype(self) -> ProtoDtype:
        """
        Dtype description as a tuple
        ``(kind, bit-width, format string, endianness)``

        Kind :

            - INT = 0
            - UINT = 1
            - FLOAT = 2
            - BOOL = 20
            - STRING = 21   # UTF-8
            - DATETIME = 22
            - CATEGORICAL = 23

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes
        -----
        - Kind specifiers are aligned with DLPack where possible
         (hence the jump to 20, leave enough room for future extension)
        - Masks must be specified as boolean with either bit width 1
         (for bit masks) or 8 (for byte masks).
        - Dtype width in bits was preferred over bytes
        - Endianness isn't too useful, but included now in case
          in the future we need to support non-native endianness
        - Went with Apache Arrow format strings over NumPy format strings
          because they're more complete from a dataframe perspective
        - Format strings are mostly useful for datetime specification,
          and for categoricals.
        - For categoricals, the format string describes the type of the
          categorical in the data buffer. In case of a separate encoding
          of the categorical (e.g. an integer to string mapping),
          this can be derived from ``self.describe_categorical``.
        - Data types not included: complex, Arrow-style null,
          binary, decimal, and nested (list, struct, map, union) dtypes.
        """
        dtype = self._col.dtype

        # For now, assume that, if the column dtype is 'O' (i.e., `object`),
        # then we have an array of strings
        if not isinstance(dtype, cudf.CategoricalDtype) and dtype.kind == "O":
            return (_DtypeKind.STRING, 8, "u", "=")

        return self._dtype_from_cudfdtype(dtype)

    def _dtype_from_cudfdtype(self, dtype) -> ProtoDtype:
        """
        See `self.dtype` for details.
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void)
        #       not handled datetime and timedelta both map to datetime
        #       (is timedelta handled?)
        _np_kinds = {
            "i": _DtypeKind.INT,
            "u": _DtypeKind.UINT,
            "f": _DtypeKind.FLOAT,
            "b": _DtypeKind.BOOL,
            "U": _DtypeKind.STRING,
            "M": _DtypeKind.DATETIME,
            "m": _DtypeKind.DATETIME,
        }
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            # Not a NumPy/CuPy dtype. Check if it's a categorical maybe
            if isinstance(dtype, cudf.CategoricalDtype):
                kind = _DtypeKind.CATEGORICAL
                # Codes and categories' dtypes are different.
                # We use codes' dtype as these are stored in the buffer.
                codes = cast(
                    cudf.core.column.CategoricalColumn, self._col
                ).codes
                dtype = codes.dtype
            else:
                raise ValueError(
                    f"Data type {dtype} not supported by exchange protocol"
                )

        if kind not in _SUPPORTED_KINDS:
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        bitwidth = dtype.itemsize * 8
        format_str = dtype.str
        endianness = dtype.byteorder if kind != _DtypeKind.CATEGORICAL else "="
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> tuple[bool, bool, dict[int, Any]]:
        """
        If the dtype is categorical, there are two options:

        - There are only values in the data buffer.
        - There is a separate dictionary-style encoding for categorical values.

        Raises TypeError if the dtype is not categorical

        Content of returned dict:

            - "is_ordered" : bool, whether the ordering of dictionary
                             indices is semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "mapping" : dict, Python-level only (e.g. ``{int: str}``).
                          None if not a dictionary-style categorical.
        """
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError(
                "`describe_categorical only works on "
                "a column with categorical dtype!"
            )
        categ_col = cast(cudf.core.column.CategoricalColumn, self._col)
        ordered = bool(categ_col.dtype.ordered)
        is_dictionary = True
        # NOTE: this shows the children approach is better, transforming
        # `categories` to a "mapping" dict is inefficient
        categories = categ_col.categories
        mapping = {ix: val for ix, val in enumerate(categories.values_host)}
        return ordered, is_dictionary, mapping

    @property
    def describe_null(self) -> tuple[int, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Kind:

            - 0 : non-nullable
            - 1 : NaN/NaT
            - 2 : sentinel value
            - 3 : bit mask
            - 4 : byte mask

        Value : if kind is "sentinel value", the actual value.
        If kind is a bit mask or a byte mask, the value (0 or 1)
        indicating a missing value.
        None otherwise.
        """
        kind = self.dtype[0]
        if self.null_count == 0:
            # there is no validity mask so it is non-nullable
            return _MaskKind.NON_NULLABLE, None

        elif kind in _SUPPORTED_KINDS:
            # currently, we return a bit mask
            return _MaskKind.BITMASK, 0

        else:
            raise NotImplementedError(
                f"Data type {self.dtype} not yet supported"
            )

    @property
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.
        """
        return self._col.null_count

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Store specific metadata of the column.
        """
        return {}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(
        self, n_chunks: int | None = None
    ) -> Iterable["_CuDFColumn"]:
        """
        Return an iterable yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        return (self,)

    def get_buffers(
        self,
    ) -> Mapping[str, tuple[_CuDFBuffer, ProtoDtype] | None]:
        """
        Return a dictionary containing the underlying buffers.

        The returned dictionary has the following contents:

            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        buffers = {}
        try:
            buffers["validity"] = self._get_validity_buffer()
        except RuntimeError:
            buffers["validity"] = None

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except RuntimeError:
            buffers["offsets"] = None

        buffers["data"] = self._get_data_buffer()

        return buffers

    def _get_validity_buffer(
        self,
    ) -> tuple[_CuDFBuffer, ProtoDtype] | None:
        """
        Return the buffer containing the mask values
        indicating missing data and the buffer's associated dtype.

        Raises RuntimeError if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null

        if null == _MaskKind.BITMASK:
            assert self._col.mask is not None
            buffer = _CuDFBuffer(
                self._col.mask, cp.uint8, allow_copy=self._allow_copy
            )
            dtype = (_DtypeKind.UINT, 8, "C", "=")
            return buffer, dtype

        elif null == _MaskKind.NAN:
            raise RuntimeError(
                "This column uses NaN as null "
                "so does not have a separate mask"
            )
        elif null == _MaskKind.NON_NULLABLE:
            raise RuntimeError(
                "This column is non-nullable so does not have a mask"
            )
        else:
            raise NotImplementedError(
                f"See {self.__class__.__name__}.describe_null method."
            )

    def _get_offsets_buffer(
        self,
    ) -> tuple[_CuDFBuffer, ProtoDtype] | None:
        """
        Return the buffer containing the offset values for
        variable-size binary data (e.g., variable-length strings)
        and the buffer's associated dtype.

        Raises RuntimeError if the data buffer does not have an associated
        offsets buffer.
        """
        if self.dtype[0] == _DtypeKind.STRING:
            offsets = self._col.children[0]
            assert (offsets is not None) and (offsets.data is not None), " "
            "offsets(.data) should not be None for string column"

            buffer = _CuDFBuffer(
                offsets.data, offsets.dtype, allow_copy=self._allow_copy
            )
            dtype = self._dtype_from_cudfdtype(offsets.dtype)
        else:
            raise RuntimeError(
                "This column has a fixed-length dtype "
                "so does not have an offsets buffer"
            )

        return buffer, dtype

    def _get_data_buffer(
        self,
    ) -> tuple[_CuDFBuffer, ProtoDtype]:
        """
        Return the buffer containing the data and
               the buffer's associated dtype.
        """
        if self.dtype[0] in (
            _DtypeKind.INT,
            _DtypeKind.UINT,
            _DtypeKind.FLOAT,
            _DtypeKind.BOOL,
        ):
            col_data = self._col
            dtype = self.dtype

        elif self.dtype[0] == _DtypeKind.CATEGORICAL:
            col_data = cast(
                cudf.core.column.CategoricalColumn, self._col
            ).codes
            dtype = self._dtype_from_cudfdtype(col_data.dtype)

        elif self.dtype[0] == _DtypeKind.STRING:
            col_data = build_column(
                data=self._col.data, dtype=np.dtype("int8")
            )
            dtype = self._dtype_from_cudfdtype(col_data.dtype)

        else:
            raise NotImplementedError(
                f"Data type {self._col.dtype} not handled yet"
            )
        assert (col_data is not None) and (col_data.data is not None), " "
        f"col_data(.data) should not be None when dtype = {dtype}"
        buffer = _CuDFBuffer(
            col_data.data, col_data.dtype, allow_copy=self._allow_copy
        )

        return buffer, dtype


class _CuDFDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``cudf.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """

    def __init__(
        self,
        df: "cudf.core.dataframe.DataFrame",
        nan_as_null: bool = True,
        allow_copy: bool = True,
    ) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `cudf.DataFrame.__dataframe__`.
        """
        self._df = df
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with
        # ``NaN`` (or ``NaT``).
        # This currently has no effect; once support for nullable extension
        # dtypes is added, this value should be propagated to columns.
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> "_CuDFDataFrame":
        """
        See the docstring of the `cudf.DataFrame.__dataframe__` for details
        """
        return _CuDFDataFrame(
            self._df, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @property
    def metadata(self):
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as cuDF-specific metadata here.
        return {"cudf.index": self._df.index}

    def num_columns(self) -> int:
        return len(self._df._column_names)

    def num_rows(self) -> int:
        return len(self._df)

    def num_chunks(self) -> int:
        return 1

    def column_names(self) -> Iterable[str]:
        return self._df._column_names

    def get_column(self, i: int) -> _CuDFColumn:
        return _CuDFColumn(
            as_column(self._df.iloc[:, i]), allow_copy=self._allow_copy
        )

    def get_column_by_name(self, name: str) -> _CuDFColumn:
        return _CuDFColumn(
            as_column(self._df[name]), allow_copy=self._allow_copy
        )

    def get_columns(self) -> Iterable[_CuDFColumn]:
        return [
            _CuDFColumn(as_column(self._df[name]), allow_copy=self._allow_copy)
            for name in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> "_CuDFDataFrame":
        if not isinstance(indices, abc.Sequence):
            raise ValueError("`indices` is not a sequence")

        return _CuDFDataFrame(self._df.iloc[:, indices])

    def select_columns_by_name(self, names: Sequence[str]) -> "_CuDFDataFrame":
        if not isinstance(names, abc.Sequence):
            raise ValueError("`names` is not a sequence")

        return _CuDFDataFrame(
            self._df.loc[:, names], self._nan_as_null, self._allow_copy
        )

    def get_chunks(
        self, n_chunks: int | None = None
    ) -> Iterable["_CuDFDataFrame"]:
        """
        Return an iterator yielding the chunks.
        """
        return (self,)


def __dataframe__(
    self, nan_as_null: bool = False, allow_copy: bool = True
) -> _CuDFDataFrame:
    """
    The public method to attach to cudf.DataFrame.

    ``nan_as_null`` is a keyword intended for the consumer to tell the
    producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
    This currently has no effect; once support for nullable extension
    dtypes is added, this value should be propagated to columns.

    ``allow_copy`` is a keyword that defines whether or not the library is
    allowed to make a copy of the data. For example, copying data would be
    necessary if a library supports strided buffers, given that this protocol
    specifies contiguous buffers.
    """
    return _CuDFDataFrame(self, nan_as_null=nan_as_null, allow_copy=allow_copy)


"""
Implementation of the dataframe exchange protocol.

Public API
----------

from_dataframe : construct a cudf.DataFrame from an input data frame which
                 implements the exchange protocol

Notes
-----

- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and
  unsafe to do in pure Python. It's more general but definitely less friendly
  than having ``to_arrow`` and ``to_numpy`` methods. So for the buffers which
  lack ``__dlpack__`` (e.g., because the column dtype isn't supported by
  DLPack), this is worth looking at again.

"""


# A typing protocol could be added later to let Mypy validate code using
# `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any


_INTS = {8: cp.int8, 16: cp.int16, 32: cp.int32, 64: cp.int64}
_UINTS = {8: cp.uint8, 16: cp.uint16, 32: cp.uint32, 64: cp.uint64}
_FLOATS = {32: cp.float32, 64: cp.float64}
_CP_DTYPES = {
    0: _INTS,
    1: _UINTS,
    2: _FLOATS,
    20: {8: bool},
    21: {8: cp.uint8},
}


def from_dataframe(
    df: DataFrameObject, allow_copy: bool = False
) -> cudf.DataFrame:
    """
    Construct a ``DataFrame`` from ``df`` if it supports the
    dataframe interchange protocol (``__dataframe__``).

    Parameters
    ----------
    df : DataFrameObject
        Object supporting dataframe interchange protocol
    allow_copy : bool
        If ``True``, allow copying of the data. If ``False``, a
        ``TypeError`` is raised if data copying is required to
        construct the ``DataFrame`` (e.g., if ``df`` lives in CPU
        memory).

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> pdf = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    >>> df = cudf.from_dataframe(pdf, allow_copy=True)
    >>> type(df)
    cudf.core.dataframe.DataFrame
    >>> df
       a  b
    0  1  x
    1  2  y
    2  3  z

    Notes
    -----
    See https://data-apis.org/dataframe-protocol/latest/index.html
    for the dataframe interchange protocol spec and API
    """
    if isinstance(df, cudf.DataFrame):
        return df

    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    df = df.__dataframe__(allow_copy=allow_copy)

    # Check number of chunks, if there's more than one we need to iterate
    if df.num_chunks() > 1:
        raise NotImplementedError("More than one chunk not handled yet")

    # We need a dict of columns here, with each column being a cudf column.
    columns = dict()
    _buffers = []  # hold on to buffers, keeps memory alive
    for name in df.column_names():
        col = df.get_column_by_name(name)

        if col.dtype[0] in (
            _DtypeKind.INT,
            _DtypeKind.UINT,
            _DtypeKind.FLOAT,
            _DtypeKind.BOOL,
        ):
            columns[name], _buf = _protocol_to_cudf_column_numeric(
                col, allow_copy
            )

        elif col.dtype[0] == _DtypeKind.CATEGORICAL:
            columns[name], _buf = _protocol_to_cudf_column_categorical(
                col, allow_copy
            )

        elif col.dtype[0] == _DtypeKind.STRING:
            columns[name], _buf = _protocol_to_cudf_column_string(
                col, allow_copy
            )

        else:
            raise NotImplementedError(
                f"Data type {col.dtype[0]} not handled yet"
            )

        _buffers.append(_buf)

    df_new = cudf.DataFrame._from_data(columns)
    df_new._buffers = _buffers
    return df_new


def _protocol_to_cudf_column_numeric(
    col, allow_copy: bool
) -> tuple[
    cudf.core.column.ColumnBase,
    Mapping[str, tuple[_CuDFBuffer, ProtoDtype] | None],
]:
    """
    Convert an int, uint, float or bool protocol column
    to the corresponding cudf column
    """
    if col.offset != 0:
        raise NotImplementedError("column.offset > 0 not handled yet")

    buffers = col.get_buffers()
    assert buffers["data"] is not None, "data buffer should not be None"
    _dbuffer, _ddtype = buffers["data"]
    _dbuffer = _ensure_gpu_buffer(_dbuffer, _ddtype, allow_copy)
    cudfcol_num = build_column(
        _dbuffer._buf,
        protocol_dtype_to_cupy_dtype(_ddtype),
    )
    return _set_missing_values(col, cudfcol_num, allow_copy), buffers


def _ensure_gpu_buffer(buf, data_type, allow_copy: bool) -> _CuDFBuffer:
    # if `buf` is a (protocol) buffer that lives on the GPU already,
    # return it as is.  Otherwise, copy it to the device and return
    # the resulting buffer.
    if buf.__dlpack_device__()[0] != _Device.CUDA:
        if allow_copy:
            dbuf = rmm.DeviceBuffer(ptr=buf.ptr, size=buf.bufsize)
            return _CuDFBuffer(
                as_buffer(dbuf, exposed=True),
                protocol_dtype_to_cupy_dtype(data_type),
                allow_copy,
            )
        else:
            raise TypeError(
                "This operation must copy data from CPU to GPU. "
                "Set `allow_copy=True` to allow it."
            )
    return buf


def _set_missing_values(
    protocol_col,
    cudf_col: cudf.core.column.ColumnBase,
    allow_copy: bool,
) -> cudf.core.column.ColumnBase:
    valid_mask = protocol_col.get_buffers()["validity"]
    if valid_mask is not None:
        null, invalid = protocol_col.describe_null
        if null == _MaskKind.BYTEMASK:
            valid_mask = _ensure_gpu_buffer(
                valid_mask[0], valid_mask[1], allow_copy
            )
            bitmask = as_column(valid_mask._buf, dtype="bool").as_mask()
            return cudf_col.set_mask(bitmask)
        elif null == _MaskKind.BITMASK:
            valid_mask = _ensure_gpu_buffer(
                valid_mask[0], valid_mask[1], allow_copy
            )
            bitmask = valid_mask._buf
            return cudf_col.set_mask(bitmask)
    return cudf_col


def protocol_dtype_to_cupy_dtype(_dtype: ProtoDtype) -> cp.dtype:
    kind = _dtype[0]
    bitwidth = _dtype[1]
    if _dtype[0] not in _SUPPORTED_KINDS:
        raise RuntimeError(f"Data type {_dtype[0]} not handled yet")

    return _CP_DTYPES[kind][bitwidth]


def _protocol_to_cudf_column_categorical(
    col, allow_copy: bool
) -> tuple[
    cudf.core.column.ColumnBase,
    Mapping[str, tuple[_CuDFBuffer, ProtoDtype] | None],
]:
    """
    Convert a categorical column to a Series instance
    """
    ordered, is_dict, categories = col.describe_categorical
    if not is_dict:
        raise NotImplementedError(
            "Non-dictionary categoricals not supported yet"
        )
    buffers = col.get_buffers()
    assert buffers["data"] is not None, "data buffer should not be None"
    codes_buffer, codes_dtype = buffers["data"]
    codes_buffer = _ensure_gpu_buffer(codes_buffer, codes_dtype, allow_copy)
    cdtype = np.dtype(protocol_dtype_to_cupy_dtype(codes_dtype))
    codes = NumericalColumn(
        data=codes_buffer._buf,
        size=None,
        dtype=cdtype,
    )
    cudfcol = CategoricalColumn(
        data=None,
        size=codes.size,
        dtype=cudf.CategoricalDtype(categories=categories, ordered=ordered),
        mask=codes.base_mask,
        offset=codes.offset,
        children=(codes,),
    )

    return _set_missing_values(col, cudfcol, allow_copy), buffers


def _protocol_to_cudf_column_string(
    col, allow_copy: bool
) -> tuple[
    cudf.core.column.ColumnBase,
    Mapping[str, tuple[_CuDFBuffer, ProtoDtype] | None],
]:
    """
    Convert a string ColumnObject to cudf Column object.
    """
    # Retrieve the data buffers
    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    assert buffers["data"] is not None, "data buffer should never be None"
    data_buffer, data_dtype = buffers["data"]
    data_buffer = _ensure_gpu_buffer(data_buffer, data_dtype, allow_copy)
    encoded_string = build_column(
        data_buffer._buf,
        protocol_dtype_to_cupy_dtype(data_dtype),
    )

    # Retrieve the offsets buffer containing the index offsets demarcating
    # the beginning and end of each string
    assert buffers["offsets"] is not None, "not possible for string column"
    offset_buffer, offset_dtype = buffers["offsets"]
    offset_buffer = _ensure_gpu_buffer(offset_buffer, offset_dtype, allow_copy)
    offsets = build_column(
        offset_buffer._buf,
        protocol_dtype_to_cupy_dtype(offset_dtype),
    )
    offsets = offsets.astype("int32")
    cudfcol_str = build_column(
        None, dtype=cp.dtype("O"), children=(offsets, encoded_string)
    )
    return _set_missing_values(col, cudfcol_str, allow_copy), buffers


def _protocol_buffer_to_cudf_buffer(protocol_buffer):
    return as_buffer(
        rmm.DeviceBuffer(
            ptr=protocol_buffer.ptr, size=protocol_buffer.bufsize
        ),
        exposed=True,
    )
