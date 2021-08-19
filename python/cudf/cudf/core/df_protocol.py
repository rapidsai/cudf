"""
Implementation of the dataframe exchange protocol.

Public API
----------

from_dataframe : construct a pandas.DataFrame from an input data frame which
                 implements the exchange protocol

Notes
-----

- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.

"""

import enum
import collections
import ctypes
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence

import cudf
import numpy as np
import cupy as cp
import pandas._testing as tm
import cudf.testing as testcase
import pytest


# A typing protocol could be added later to let Mypy validate code using
# `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any


def from_dataframe(df : DataFrameObject, copy: bool = False) :
    """
    Construct a cudf DataFrame from ``df`` if it supports ``__dataframe__``
    """
    if isinstance(df, cudf.DataFrame):
        return df

    if not hasattr(df, '__dataframe__'):
        raise ValueError("`df` does not support __dataframe__")

    return _from_dataframe(df.__dataframe__(), copy=copy)


def _from_dataframe(df : DataFrameObject, copy: bool = False) :
    """
    Note: not all cases are handled yet, only ones that can be implemented with
    only Pandas. Later, we need to implement/test support for categoricals,
    bit/byte masks, chunk handling, etc.
    """
    # Check number of chunks, if there's more than one we need to iterate
    if df.num_chunks() > 1:
        raise NotImplementedError

    # We need a dict of columns here, with each column being a numpy array (at
    # least for now, deal with non-numpy dtypes later).
    columns = dict()
    _k = _DtypeKind
    for name in df.column_names():
        col = df.get_column_by_name(name)
        if col.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            # Simple numerical or bool dtype, turn into numpy array
            columns[name] = convert_column_to_cupy_ndarray(col, copy=copy)
        elif col.dtype[0] == _k.CATEGORICAL:
            columns[name] = convert_categorical_column(col, copy=copy)
            names = df.column_names()
        else:
            raise NotImplementedError(f"Data type {col.dtype[0]} not handled yet")
    
    return cudf.DataFrame(columns)



class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21   # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


def convert_column_to_cupy_ndarray(col : ColumnObject, copy : bool = False) -> np.ndarray:
    """
    Convert an int, uint, float or bool column to a numpy array
    """
    if col.offset != 0:
        raise NotImplementedError("column.offset > 0 not handled yet")

    if col.describe_null[0] not in (0, 1):
        raise NotImplementedError("Null values represented as masks or "
                                  "sentinel values not handled yet")

    _buffer, _dtype = col.get_data_buffer()
    return buffer_to_cupy_ndarray(_buffer, _dtype, copy=copy)

def buffer_to_cupy_ndarray(_buffer, _dtype, copy : bool = False) -> cp.ndarray:
    if _buffer.__dlpack_device__()[0] == 2: # dataframe is on GPU/CUDA
        x = cp.fromDlpack(_buffer.__dlpack__())

    elif copy == False:
        raise TypeError("This operation must copy data from CPU to GPU. Set `copy=True` to allow it.")

    else:
        x = _copy_buffer_to_gpu(_buffer, _dtype)

    return x


def _copy_buffer_to_gpu(_buffer, _dtype):
    # Handle the dtype
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = _DtypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
        raise RuntimeError("Not a boolean, integer or floating-point dtype")

    _ints = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    _uints = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _floats = {32: np.float32, 64: np.float64}
    _np_dtypes = {0: _ints, 1: _uints, 2: _floats, 20: {8: bool}}
    column_dtype = _np_dtypes[kind][bitwidth]

    # No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(_buffer.ptr, ctypes.POINTER(ctypes_type))

    # NOTE: `x` does not own its memory, so the caller of this function must
    #       either make a copy or hold on to a reference of the column or
    #       buffer! (not done yet, this is pretty awful ...)
    x = np.ctypeslib.as_array(data_pointer,
                              shape=(_buffer.bufsize // (bitwidth//8),))
    return cp.array(x, dtype=column_dtype)


def convert_categorical_column(col : ColumnObject, copy:bool=False) :
    """
    Convert a categorical column to a Series instance
    """
    

    ordered, is_dict, mapping = col.describe_categorical
    if not is_dict:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')

    # If you want to cheat for testing (can't use `_col` in real-world code):
    #    categories = col._col.values.categories.values
    #    codes = col._col.values.codes
    categories = cp.asarray(list(mapping.values()))
    codes_buffer, codes_dtype = col.get_data_buffer()
    codes = buffer_to_cupy_ndarray(codes_buffer, codes_dtype, copy=copy)
    values = categories[codes]

    # Seems like Pandas can only construct with non-null values, so need to
    # null out the nulls later
    cat = cudf.CategoricalIndex(values, categories=categories, ordered=ordered)
    series = cudf.Series(cat)
    null_kind = col.describe_null[0]
    if null_kind == 2:  # sentinel value
        sentinel = col.describe_null[1]
        series[codes == sentinel] = None
    else:
        raise NotImplementedError("Only categorical columns with sentinel "
                                  "value supported at the moment")

    return series


def __dataframe__(self, nan_as_null : bool = False) -> dict:
    """
    , target_device:str = 'gpu'
    The public method to attach to cudf.DataFrame

    We'll attach it via monkeypatching here for demo purposes. If Pandas adopt
    the protocol, this will be a regular method on pandas.DataFrame.

    ``nan_as_null`` is a keyword intended for the consumer to tell the
    producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
    This currently has no effect; once support for nullable extension
    dtypes is added, this value should be propagated to columns.

    ``target_device`` specifies the device where the returned dataframe protocol
    object will live. Only `cpu` and `gpu` are supported for now.
    """
    # if target_device not in ['cpu', 'gpu']:
    #     raise TypeError (f'Device {device} support not handle.')

    # if device == 'cpu':
    #     raise TypeError("This operation will copy data from GPU to CPU. Set `copy=True` to allow it.")


    return _CuDFDataFrame(self, nan_as_null=nan_as_null)


# Monkeypatch the Pandas DataFrame class to support the interchange protocol
# cudf.DataFrame.__dataframe__ = __dataframe__


# Implementation of interchange protocol
# --------------------------------------

class _CuDFBuffer:

    """
    Data in the buffer is guaranteed to be contiguous in memory.
    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.
    This distinction is useful to support both data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    

    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, x : cp.ndarray) -> None:
        """
        Handle only regular columns (= cupy arrays) for now.
        """
        if not x.strides == (x.dtype.itemsize,):
            # Array is not contiguous - this is possible to get in Pandas,
            # there was some discussion on whether to support it. Some extra
            # complexity for libraries that don't support it (e.g. Arrow),
            # but would help with cupy-based libraries like CuDF.
            raise RuntimeError("Design needs fixing - non-contiguous buffer")

        # Store the numpy array in which the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes
        """
        return self._x.data.mem.size
        # return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer
        """
        # return self._x.data.mem.ptr
        return self._x.__cuda_array_interface__['data'][0]

    def __dlpack__(self):

        """
        Produce DLPack capsule (see array API standard).
        Raises:
            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented
        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        

        DLPack implemented in CuPy
        """
        try: 
            res = self._x.toDlpack()
        except ValueError:
            raise TypeError(f'dtype {self._x.dtype} unsupported by `dlpack`')

        return res

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:

        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack. Enum members are::
            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10
        Note: must be implemented even if ``__dlpack__`` is not.
        

        Device type and device ID for where the data in the buffer resides.
        """
        class Device(enum.IntEnum):
            CUDA = 2

        return (Device.CUDA, self._x.device.id)

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

    A column can contain one or more chunks. Each chunk can contain either one
    or two buffers - one data buffer and (depending on null representation) it
    may have a mask buffer.

     TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.
         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.
         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.
    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.
         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?


    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.

    """

    def __init__(self, column, nan_as_null=False) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if not isinstance(column, cudf.Series):
            raise NotImplementedError(f"Columns of type {type(column)} not handled yet")

        # Store the column as a private attribute
        self._col = column
        self._nan_as_null = nan_as_null

    @property
    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.
        """
        return self._col.size

    @property
    def offset(self) -> int:
        """
        Offset of first element. Always zero.
        TODO: check `Always zero (in case of cudf)?`

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
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
        return self._dtype_from_cudfdtype(dtype)

    def _dtype_from_cudfdtype(self, dtype) -> Tuple[enum.IntEnum, int, str, str]:
        """
        See `self.dtype` for details
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void) not handled
        #       datetime and timedelta both map to datetime (is timedelta handled?)
        _k = _DtypeKind
        _np_kinds = {'i': _k.INT, 'u': _k.UINT, 'f': _k.FLOAT, 'b': _k.BOOL,
                     'U': _k.STRING,
                     'M': _k.DATETIME, 'm': _k.DATETIME}
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            # Not a NumPy dtype. Check if it's a categorical maybe
            # CuPy uses NumPy dtypes.
            if isinstance(dtype, cudf.CategoricalDtype):
                kind = 23
                # Codes and categorical values dtypes are different.
                # We use codes' dtype as these are stored in the buffer. 
                dtype = self._col.cat.codes.dtype
            else:
                raise ValueError(f"Data type {dtype} not supported by exchange"
                                 "protocol")

        if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.CATEGORICAL):
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        bitwidth = dtype.itemsize * 8
        format_str = dtype.str
        endianness = dtype.byteorder if not kind == _k.CATEGORICAL else '='
        return (kind, bitwidth, format_str, endianness)


    @property
    def describe_categorical(self) -> Dict[str, Any]:
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


        TBD: are there any other in-memory representations that are needed?
        """
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError("`describe_categorical only works on a column with "
                            "categorical dtype!")

        ordered = self._col.dtype.ordered
        is_dictionary = True
        # NOTE: this shows the children approach is better, transforming
        # `categories` to a "mapping" dict is inefficient
        codes = self._col.cat.codes  # ndarray, length `self.size`
        # categories.values is ndarray of length n_categories
        categories = self._col.cat.categories
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

        Value : if kind is "sentinel value", the actual value. None otherwise.
        """
        _k = _DtypeKind
        kind = self.dtype[0]
        value = None
        if kind == _k.FLOAT:
            null = 1  # np.nan
        elif kind == _k.DATETIME:
            null = 1  # np.datetime64('NaT')
        elif kind in (_k.INT, _k.UINT, _k.BOOL):
            # TODO: check if extension dtypes are used once support for them is
            #       implemented in this procotol code
            null = 0  # integer and boolean dtypes are non-nullable
        elif kind == _k.CATEGORICAL:
            # Null values for categoricals are stored as `-1` sentinel values
            # in the category date (e.g., `col.cat.codes` is uint8 np.ndarray at least)
            null = 2
            value = -1
        else:
            raise NotImplementedError(f'Data type {self.dtype} not yet supported')

        return null, value

    @property
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.

        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        return self._col.isna().sum()

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.

        TBC: Seems like chunks are used for parallel computation purpose in cudf:`apply_chunks`.
        """
        return 1

    def get_chunks(self, n_chunks : Optional[int] = None) -> Iterable['_CuDFColumn']:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        return (self,)

    def get_data_buffer(self) -> Tuple[_CuDFBuffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data.
        """
        _k = _DtypeKind
        if self.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            buffer = _CuDFBuffer(cp.array(self._col.to_gpu_array(), copy=False))
            dtype = self.dtype
        elif self.dtype[0] == _k.CATEGORICAL:
            _, value = self.describe_null
            codes = self._col.cat.codes
            # handling null/NaN
            buffer = _CuDFBuffer(cp.array(codes.fillna(100), copy=False))
            dtype = self._dtype_from_cudfdtype(codes.dtype)
        else:
            raise NotImplementedError(f"Data type {self._col.dtype} not handled yet")

        return buffer, dtype

    def get_mask(self) -> _CuDFBuffer:
        """
        Return the buffer containing the mask values indicating missing data.

        Raises RuntimeError if null representation is not a bit or byte mask.
        """
        null, value = self.describe_null
        if null == 0:
            msg = "This column is non-nullable so does not have a mask"
        elif null == 1:
            msg = "This column uses NaN as null so does not have a separate mask"
        else:
            raise NotImplementedError('See self.describe_null')

        raise RuntimeError(msg)

    # def get_children(self) -> Iterable[Column]:
    #     """
    #     Children columns underneath the column, each object in this iterator
    #     must adhere to the column specification
    #     """
    #     pass

class _CuDFDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``cudf.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """
    def __init__(self, df, nan_as_null : bool = False) -> None:
        """
        , device:str = 'gpu'
        Constructor - an instance of this (private) class is returned from
        `cudf.DataFrame.__dataframe__`.
        """
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
        # This currently has no effect; once support for nullable extension
        # dtypes is added, this value should be propagated to columns.
        #
        # ``device`` indicates the target device for the data.
        self._nan_as_null = nan_as_null
        self._df = df

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df)

    def num_chunks(self) -> int:
        return 1

    def column_names(self) -> Iterable[str]:
        return self._df.columns.tolist()

    def get_column(self, i: int) -> _CuDFColumn:
        return _CuDFColumn(self._df.iloc[:, i])

    def get_column_by_name(self, name: str) -> _CuDFColumn:
        return _CuDFColumn(self._df[name])

    def get_columns(self) -> Iterable[_CuDFColumn]:
        return [_CuDFColumn(self._df[name]) for name in self._df.columns]

    def select_columns(self, indices: Sequence[int]) -> '_CuDFDataFrame':
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return _CuDFDataFrame(self._df.iloc[:, indices])
    
    def select_columns_by_name(self, names: Sequence[str]) -> '_CuDFDataFrame':
        """
            Create a new DataFrame by selecting a subset of columns by name.

            Don't use pandas.DataFrame `xs` method as :
            def xs(self, key, axis=0, level=None, drop_level: bool_t = True):
            
            Return cross-section from the Series/DataFrame.

            This method takes a `key` argument to select data at a particular
            level of a MultiIndex.
        """
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return _CuDFDataFrame(self._df.loc[:, names])

    def get_chunks(self, n_chunks : Optional[int] = None) -> Iterable['_CuDFDataFrame']:
        """
        Return an iterator yielding the chunks.
        """
        return (self,)
