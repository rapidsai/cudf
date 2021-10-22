import enum
import collections
import ctypes
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence

import cudf
from cudf.core.column import as_column, build_column, build_categorical_column
from cudf.core.buffer import Buffer
import numpy as np
import cupy as cp
from numba import cuda





# Implementation of interchange protocol classes
# ----------------------------------------------

class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21   # UTF-8
    DATETIME = 22
    CATEGORICAL = 23

class Device(enum.IntEnum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class _CuDFBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, buf : Buffer, cudf_dtype, allow_copy : bool = True) -> None:
        """
        Use cudf Buffer object.
        """
        # Store the cudf buffer where the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._buf = buf
        self._cudf_dtype =  cudf_dtype
        self._allow_copy = allow_copy

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._buf.nbytes

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._buf.ptr
        
    def __dlpack__(self):
        """
        DLPack not implemented in NumPy yet, so leave it out here.
        """
        try: 
            cudarray = cuda.as_cuda_array(self._buf).view(self._cudf_dtype)
            res = cp.asarray(cudarray).toDlpack()

        except ValueError:
            raise TypeError(f'dtype {self._cudf_dtype} unsupported by `dlpack`')

        return res

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
        return (Device.CUDA, cp.asarray(self._buf).device.id)

    def __repr__(self) -> str:
        return 'CuDFBuffer(' + str({'bufsize': self.bufsize,
                                      'ptr': self.ptr,
                                      'dlpack': self.__dlpack__(),
                                      'device': self.__dlpack_device__()[0].name}
                                      ) + ')'

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

    def __init__(self, column,
                 nan_as_null : bool = True, 
                 allow_copy: bool = True) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if not isinstance(column, cudf.Series):
            raise NotImplementedError("Columns of type {} not handled "
                                      "yet".format(type(column)))

        # Store the column as a private attribute
        self._col = as_column(column)
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    @property
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
    def dtype(self) -> Tuple[enum.IntEnum, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``

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

        Notes:

            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for bit
              masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the future
              we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary, decimal,
              and nested (list, struct, map, union) dtypes.
        """
        dtype = self._col.dtype

        # For now, assume that, if the column dtype is 'O' (i.e., `object`), then we have an array of strings
        if not isinstance(dtype, cudf.CategoricalDtype) and dtype.kind == 'O':
            return (_DtypeKind.STRING, 8, 'u', '=')

        return self._dtype_from_cudfdtype(dtype)

    def _dtype_from_cudfdtype(self, dtype) -> Tuple[enum.IntEnum, int, str, str]:
        """
        See `self.dtype` for details.
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void) not handled
        #       datetime and timedelta both map to datetime (is timedelta handled?)
        _k = _DtypeKind
        _np_kinds = {"i": _k.INT, "u": _k.UINT, "f": _k.FLOAT, "b": _k.BOOL,
                     "U": _k.STRING,
                     "M": _k.DATETIME, "m": _k.DATETIME}
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            # Not a NumPy/CuPy dtype. Check if it's a categorical maybe
            if isinstance(dtype, cudf.CategoricalDtype):
                kind = _k.CATEGORICAL
                # Codes and categories' dtypes are different.
                # We use codes' dtype as these are stored in the buffer. 
                dtype = self._col.codes.dtype
            else:
                raise ValueError(f"Data type {dtype} not supported by exchange"
                                 "protocol")

        if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.CATEGORICAL, _k.STRING):
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        bitwidth = dtype.itemsize * 8
        format_str = dtype.str
        endianness = dtype.byteorder if not kind == _k.CATEGORICAL else '='
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> Tuple[Any, bool, Dict[int, Any]]:
        """
        If the dtype is categorical, there are two options:

        - There are only values in the data buffer.
        - There is a separate dictionary-style encoding for categorical values.

        Raises RuntimeError if the dtype is not categorical

        Content of returned dict:

            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "mapping" : dict, Python-level only (e.g. ``{int: str}``).
                          None if not a dictionary-style categorical.
        """
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError("`describe_categorical only works on a column with "
                            "categorical dtype!")

        ordered = self._col.dtype.ordered
        is_dictionary = True
        # NOTE: this shows the children approach is better, transforming
        # `categories` to a "mapping" dict is inefficient
        codes = self._col.codes  # ndarray, length `self.size`
        # categories.values is ndarray of length n_categories
        categories = self._col.categories
        mapping = {ix: val for ix, val in enumerate(categories.values_host)}
        return ordered, is_dictionary, mapping

    @property
    def describe_null(self) -> Tuple[int, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Kind:

            - 0 : non-nullable
            - 1 : NaN/NaT
            - 2 : sentinel value
            - 3 : bit mask
            - 4 : byte mask

        Value : if kind is "sentinel value", the actual value.  If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value. None
        otherwise.
        """
        if self.null_count == 0:
            # there is no validity mask in this case
            # so making it non-nullable (hackingly)
            null = 0
            value = None
        else :
            _k = _DtypeKind
            kind = self.dtype[0]
            # bit mask is universally used in cudf for missing
            if kind in (_k.INT, _k.UINT, _k.FLOAT, _k.CATEGORICAL,
                        _k.BOOL, _k.STRING, _k.DATETIME):
                null = 3
                value = 0
            else:
                raise NotImplementedError(f"Data type {self.dtype} not yet supported")

        return null, value

    @property
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.
        """
        return self._col.isna().sum()

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Store specific metadata of the column.
        """
        return {}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(self, n_chunks : Optional[int] = None) -> Iterable['_CuDFColumn']:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        return (self,)

    def get_buffers(self) -> Dict[str, Any]:
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
        buffers["data"] = self._get_data_buffer()
        try:
            buffers["validity"] = self._get_validity_buffer()
        except:
            buffers["validity"] = None

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except:
            buffers["offsets"] = None

        return buffers

    def _get_data_buffer(self) -> Tuple[_CuDFBuffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
        _k = _DtypeKind
        invalid = self.describe_null[1]
        if self.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            buffer = _CuDFBuffer(self._col.data, self._col.dtype, 
                                 allow_copy=self._allow_copy)
            dtype = self.dtype

        elif self.dtype[0] == _k.CATEGORICAL:
            codes = self._col.codes
            buffer = _CuDFBuffer(self._col.codes.data, self._col.codes.dtype, 
                                 allow_copy=self._allow_copy)
            dtype = self._dtype_from_cudfdtype(codes.dtype)

        elif self.dtype[0] == _k.STRING:
            encoded_string = self._col.children[1]
            buffer = _CuDFBuffer(encoded_string.data, encoded_string.dtype, allow_copy=self._allow_copy)
            dtype = self._dtype_from_cudfdtype(encoded_string.dtype) 
            # dtype = (_k.STRING, 8, "u", "=") 

        else:
            raise NotImplementedError(f"Data type {self._col.dtype} not handled yet")

        return buffer, dtype

    def _get_validity_buffer(self) -> Tuple[_CuDFBuffer, Any]:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.

        Raises RuntimeError if null representation is not a bit or byte mask.
        """
        
        null, invalid = self.describe_null
        if null == 3:
            _k = _DtypeKind
            if self.dtype[0] == _k.CATEGORICAL:
                buffer = _CuDFBuffer(self._col.codes._get_mask_as_column().data, cp.uint8, 
                                     allow_copy=self._allow_copy)
            else:
                buffer = _CuDFBuffer(self._col._get_mask_as_column().data, cp.uint8,
                                     allow_copy=self._allow_copy)
            dtype = (_k.UINT, 8, "C", "=")
            return buffer, dtype

        elif null == 1:
            msg = "This column uses NaN as null so does not have a separate mask"
        elif null == 0:   
            msg = "This column is non-nullable so does not have a mask"
        else:
            raise NotImplementedError("See self.describe_null")

        raise RuntimeError(msg)

    def _get_offsets_buffer(self) -> Tuple[_CuDFBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.

        Raises RuntimeError if the data buffer does not have an associated
        offsets buffer.
        """
        _k = _DtypeKind
        if self.dtype[0] == _k.STRING:
            offsets = self._col.children[0]
            buffer = _CuDFBuffer(offsets.data, offsets.dtype, allow_copy=self._allow_copy)
            dtype = self._dtype_from_cudfdtype(offsets.dtype) 
        else:
            raise RuntimeError("This column has a fixed-length dtype so does not have an offsets buffer")

        return buffer, dtype

class _CuDFDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``cudf.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """
    def __init__(self, df, nan_as_null : bool = True,
                 allow_copy : bool = True) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `cudf.DataFrame.__dataframe__`.
        """
        self._df = df
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
        # This currently has no effect; once support for nullable extension
        # dtypes is added, this value should be propagated to columns.
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    @property
    def metadata(self):
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as Pandas-specific metadata here.
        return {"cudf.index": self._df.index}

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df)

    def num_chunks(self) -> int:
        return 1

    def column_names(self) -> Iterable[str]:
        return self._df.columns.tolist()

    def get_column(self, i: int) -> _CuDFColumn:
        return _CuDFColumn(
            self._df.iloc[:, i], allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _CuDFColumn:
        return _CuDFColumn(
            self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_CuDFColumn]:
        return [_CuDFColumn(self._df[name], allow_copy=self._allow_copy)
                for name in self._df.columns]

    def select_columns(self, indices: Sequence[int]) -> '_CuDFDataFrame':
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return _CuDFDataFrame(self._df.iloc[:, indices])

    def select_columns_by_name(self, names: Sequence[str]) -> '_CuDFDataFrame':
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return _CuDFDataFrame(self._df.loc[:, names], self._nan_as_null,
                                self._allow_copy)

    def get_chunks(self, n_chunks : Optional[int] = None) -> Iterable['_CuDFDataFrame']:
        """
        Return an iterator yielding the chunks.
        """
        return (self,)

"""
Implementation of the dataframe exchange protocol.

Public API
----------

from_dataframe : construct a cudf.DataFrame from an input data frame which
                 implements the exchange protocol

Notes
-----

- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.

"""


# A typing protocol could be added later to let Mypy validate code using
# `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any


_INTS = {8: cp.int8, 16: cp.int16, 32: cp.int32, 64: cp.int64}
_UINTS = {8: cp.uint8, 16: cp.uint16, 32: cp.uint32, 64: cp.uint64}
_FLOATS = {32: cp.float32, 64: cp.float64}
_CP_DTYPES = {0: _INTS, 1: _UINTS, 2: _FLOATS, 20: {8: bool}}


def from_dataframe(df : DataFrameObject, allow_copy: bool = False) -> _CuDFDataFrame :
    """
    Construct a cudf DataFrame from ``df`` if it supports ``__dataframe__``
    """
    if isinstance(df, cudf.DataFrame):
        return df

    if not hasattr(df, '__dataframe__'):
        raise ValueError("`df` does not support __dataframe__")

    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy))


def _from_dataframe(df : DataFrameObject) :
    """
    Create a cudf DataFrame object from DataFrameObject.
    """
    # Check number of chunks, if there's more than one we need to iterate
    if df.num_chunks() > 1:
        raise NotImplementedError("More than one chunk not handled yet")

    # We need a dict of columns here, with each column being a cudf column column.
    columns = dict()
    _k = _DtypeKind
    _buffers = []  # hold on to buffers, keeps memory alive
    for name in df.column_names():
        col = df.get_column_by_name(name)

        if col.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            columns[name], _buf = _protocol_column_to_cudf_column_numeric(col)

        elif col.dtype[0] == _k.CATEGORICAL:
            columns[name], _buf = _protocol_column_to_cudf_column_categorical(col)

        elif col.dtype[0] == _k.STRING:
            columns[name], _buf = _protocol_column_to_cudf_column_string(col)
            
        else:
            raise NotImplementedError(f"Data type {col.dtype[0]} not handled yet")
        
        _buffers.append(_buf)

    df_new = cudf.DataFrame._from_data(columns)
    df_new._buffers = _buffers
    return df_new


def _protocol_column_to_cudf_column_numeric(col:ColumnObject):
    """
    Convert an int, uint, float or bool protocol column to the corresponding cudf column
    """
    if col.offset != 0:
        raise NotImplementedError("column.offset > 0 not handled yet")

    _dbuffer, _ddtype = col.get_buffers()['data']
    _check_data_is_on_gpu(_dbuffer)
    dcol = build_column(Buffer(_dbuffer.ptr, _dbuffer.bufsize), 
                        protocol_dtypes_to_cupy_dtype(_ddtype))        
    return _set_missing_values(col, dcol), _dbuffer


def _check_data_is_on_gpu(buffer):
    if buffer.__dlpack_device__()[0] != Device.CUDA and not buffer._allow_copy:
        raise TypeError("This operation must copy data from CPU to GPU."
                            "Set `allow_copy=True` to allow it.")

def _set_missing_values(protocol_col, cudf_col):
    null_kind, null_value = protocol_col.describe_null
    if  null_kind != 0:
        assert null_kind == 3, f"cudf supports only bit mask, null_kind should be 3 ." 
        _mask_buffer, _mask_dtype = protocol_col.get_buffers()["validity"]
        bitmask = cp.asarray(Buffer(_mask_buffer.ptr, _mask_buffer.bufsize), cp.bool8) 
        cudf_col[~bitmask] = None

    return cudf_col

def protocol_dtypes_to_cupy_dtype(_dtype):
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = _DtypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL,_k.CATEGORICAL,
                         _k.STRING, _k.DATETIME):
        raise RuntimeError(f"Data type {_dtype[0]} not handled yet")
   
    return _CP_DTYPES[kind][bitwidth]

def _protocol_column_to_cudf_column_categorical(col : ColumnObject) :
    """
    Convert a categorical column to a Series instance
    """
    ordered, is_dict, mapping = col.describe_categorical
    if not is_dict:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')

    categories = as_column(mapping.values())
    codes_buffer, codes_dtype = col.get_buffers()['data']
    _check_data_is_on_gpu(codes_buffer)
    cdtype = protocol_dtypes_to_cupy_dtype(codes_dtype)
    codes = build_column(Buffer(codes_buffer.ptr, codes_buffer.bufsize), cdtype)
    
    col1 = build_categorical_column(categories=categories,codes=codes,mask=codes.base_mask,
                                    size=codes.size,ordered=ordered)

    return _set_missing_values(col, col1), codes_buffer


def _protocol_column_to_cudf_column_string(col : ColumnObject) :
    """
    Convert a string ColumnObject to cudf Column object.
    """
    # Retrieve the data buffers
    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    dbuffer, bdtype = buffers["data"]
    _check_data_is_on_gpu(dbuffer)
    encoded_string = build_column(Buffer(dbuffer.ptr, dbuffer.bufsize),
                        protocol_dtypes_to_cupy_dtype(bdtype)
                        )

    # Retrieve the offsets buffer containing the index offsets demarcating the beginning and end of each string
    obuffer, odtype = buffers["offsets"]
    offsets = build_column(Buffer(obuffer.ptr, obuffer.bufsize), 
                           protocol_dtypes_to_cupy_dtype(odtype)
                           )
    
    col_str = build_column(None, dtype=cp.dtype('O'), children=(offsets, encoded_string))

    return _set_missing_values(col, col_str), buffers



def __dataframe__(self, nan_as_null : bool = False,
                  allow_copy : bool = True) -> dict:
    """
    The public method to attach to cudf.DataFrame.

    We'll attach it via monkey-patching here for demo purposes. If Pandas adopts
    the protocol, this will be a regular method on pandas.DataFrame.

    ``nan_as_null`` is a keyword intended for the consumer to tell the
    producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
    This currently has no effect; once support for nullable extension
    dtypes is added, this value should be propagated to columns.

    ``allow_copy`` is a keyword that defines whether or not the library is
    allowed to make a copy of the data. For example, copying data would be
    necessary if a library supports strided buffers, given that this protocol
    specifies contiguous buffers.
    Currently, if the flag is set to ``False`` and a copy is needed, a
    ``RuntimeError`` will be raised.
    """
    return _CuDFDataFrame(
        self, nan_as_null=nan_as_null, allow_copy=allow_copy)

