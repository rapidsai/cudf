"""
Provides base classes and utils for implementing type-specific logical
view of Columns.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from numba import cuda, njit

from librmm_cffi import librmm as rmm
import nvstrings

from cudf.dataframe.buffer import Buffer
from cudf.dataframe.column import Column
from cudf.utils import utils, cudautils
from cudf.utils.utils import buffers_from_pyarrow
from cudf.bindings.cudf_cpp import np_to_pa_dtype
from cudf.bindings.stream_compaction import (cpp_drop_nulls,
                                             cpp_apply_boolean_mask)

import warnings
import cudf.bindings.copying as cpp_copying


class TypedColumnBase(Column):
    """Base class for all typed column
    e.g. NumericalColumn, CategoricalColumn

    This class provides common operations to implement logical view and
    type-based operations for the column.

    Notes
    -----
    Not designed to be instantiated directly.  Instantiate subclasses instead.
    """
    def __init__(self, **kwargs):
        dtype = kwargs.pop('dtype')
        super(TypedColumnBase, self).__init__(**kwargs)
        # Logical dtype
        self._dtype = pd.api.types.pandas_dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    def is_type_equivalent(self, other):
        """Is the logical type of the column equal to the other column.
        """
        mine = self._replace_defaults()
        theirs = other._replace_defaults()

        def remove_base(dct):
            # removes base attributes in the phyiscal layer.
            basekeys = Column._replace_defaults(self).keys()
            for k in basekeys:
                del dct[k]

        remove_base(mine)
        remove_base(theirs)

        return type(self) == type(other) and mine == theirs

    def _replace_defaults(self):
        params = super(TypedColumnBase, self)._replace_defaults()
        params.update(dict(dtype=self._dtype))
        return params

    def _mimic_inplace(self, result, inplace=False):
        """
        If `inplace=True`, used to mimic an inplace operation
        by replacing data in ``self`` with data in ``result``.

        Otherwise, returns ``result`` unchanged.
        """
        if inplace:
            self._data = result._data
            self._mask = result._mask
            self._null_count = result._null_count
        else:
            return result

    def argsort(self, ascending):
        _, inds = self.sort_by_values(ascending=ascending)
        return inds

    def sort_by_values(self, ascending):
        raise NotImplementedError

    def find_and_replace(self, to_replace, values):
        raise NotImplementedError

    def dropna(self):
        dropped_col = cpp_drop_nulls(self)
        return self.replace(data=dropped_col.data, mask=None, null_count=0)

    def apply_boolean_mask(self, mask):
        mask = as_column(mask, dtype="bool")
        data = cpp_apply_boolean_mask(self, mask)
        return self.replace(data=data.data, mask=data.mask)

    def fillna(self, fill_value, inplace):
        raise NotImplementedError


def column_empty_like(column, dtype=None, masked=False, newsize=None):
    """Allocate a new column like the given *column*
    """
    if dtype is None:
        dtype = column.dtype
    row_count = len(column) if newsize is None else newsize
    categories = None
    if pd.api.types.is_categorical_dtype(dtype):
        categories = column.cat().categories
        dtype = column.data.dtype
    return column_empty(row_count, dtype, masked, categories=categories)


def column_empty(row_count, dtype, masked, categories=None):
    """Allocate a new column like the given row_count and dtype.
    """
    dtype = pd.api.types.pandas_dtype(dtype)

    if masked:
        mask = cudautils.make_mask(row_count)
        cudautils.fill_value(mask, 0)
    else:
        mask = None

    if (
        categories is not None
        or pd.api.types.is_categorical_dtype(dtype)
    ):
        mem = rmm.device_array((row_count,), dtype=dtype)
        data = Buffer(mem)
        dtype = 'category'
    elif dtype.kind in 'OU':
        if row_count == 0:
            data = nvstrings.to_device([])
        else:
            mem = rmm.device_array((row_count,), dtype='float64')
            data = nvstrings.dtos(mem,
                                  len(mem),
                                  nulls=mask,
                                  bdevmem=True)
    else:
        mem = rmm.device_array((row_count,), dtype=dtype)
        data = Buffer(mem)

    if mask is not None:
        mask = Buffer(mask)

    from cudf.dataframe.columnops import build_column
    return build_column(data,
                        dtype,
                        mask,
                        categories)


def column_empty_like_same_mask(column, dtype):
    """Create a new empty Column with the same length and the same mask.

    Parameters
    ----------
    dtype : np.dtype like
        The dtype of the data buffer.
    """
    data = rmm.device_array(shape=len(column), dtype=dtype)
    params = dict(data=Buffer(data))
    if column.has_null_mask:
        params.update(mask=column.nullmask)
    return Column(**params)


def column_select_by_boolmask(column, boolmask):
    """Select by a boolean mask to a column.

    Returns (selected_column, selected_positions)
    """
    from cudf.dataframe.numerical import NumericalColumn
    assert column.null_count == 0  # We don't properly handle the boolmask yet
    boolbits = cudautils.compact_mask_bytes(boolmask.to_gpu_array())
    indices = cudautils.arange(len(boolmask))
    _, selinds = cudautils.copy_to_dense(indices, mask=boolbits)
    _, selvals = cudautils.copy_to_dense(column.data.to_gpu_array(),
                                         mask=boolbits)

    selected_values = column.replace(data=Buffer(selvals))
    selected_index = Buffer(selinds)
    return selected_values, NumericalColumn(data=selected_index,
                                            dtype=selected_index.dtype)


def column_select_by_position(column, positions):
    """Select by a series of dtype int64 indicating positions.

    Returns (selected_column, selected_positions)
    """
    from cudf.dataframe.numerical import NumericalColumn

    pos_ary = positions.data.to_gpu_array()
    selected_values = cpp_copying.apply_gather_column(column, pos_ary)
    selected_index = Buffer(pos_ary)

    return selected_values, NumericalColumn(data=selected_index,
                                            dtype=selected_index.dtype)


def build_column(buffer, dtype, mask=None, categories=None):
    from cudf.dataframe import numerical, categorical, datetime, string
    if pd.api.types.is_categorical_dtype(dtype):
        return categorical.CategoricalColumn(data=buffer,
                                             dtype='categorical',
                                             categories=categories,
                                             ordered=False,
                                             mask=mask)
    elif np.dtype(dtype).type == np.datetime64:
        return datetime.DatetimeColumn(data=buffer,
                                       dtype=np.dtype(dtype),
                                       mask=mask)
    elif np.dtype(dtype).type in (np.object_, np.str_):
        if not isinstance(buffer, nvstrings.nvstrings):
            raise TypeError
        return string.StringColumn(data=buffer)
    else:
        return numerical.NumericalColumn(data=buffer,
                                         dtype=dtype,
                                         mask=mask)


def as_column(arbitrary, nan_as_null=True, dtype=None):
    """Create a Column from an arbitrary object

    Currently support inputs are:

    * ``Column``
    * ``Buffer``
    * ``Series``
    * ``Index``
    * numba device array
    * cuda array interface
    * numpy array
    * pyarrow array
    * pandas.Categorical

    Returns
    -------
    result : subclass of TypedColumnBase
        - CategoricalColumn for pandas.Categorical input.
        - DatetimeColumn for datetime input
        - NumericalColumn for all other inputs.
    """
    from cudf.dataframe import numerical, categorical, datetime, string
    from cudf.dataframe.series import Series
    from cudf.dataframe.index import Index

    if isinstance(arbitrary, Column):
        categories = None
        if hasattr(arbitrary, "categories"):
            categories = arbitrary.categories
        data = build_column(
            arbitrary.data,
            arbitrary.dtype,
            mask=arbitrary.mask,
            categories=categories
        )

    elif isinstance(arbitrary, Series):
        data = arbitrary._column

    elif isinstance(arbitrary, Index):
        data = arbitrary._values

    elif isinstance(arbitrary, Buffer):
        data = numerical.NumericalColumn(data=arbitrary, dtype=arbitrary.dtype)

    elif isinstance(arbitrary, nvstrings.nvstrings):
        data = string.StringColumn(data=arbitrary)

    elif cuda.devicearray.is_cuda_ndarray(arbitrary):
        data = as_column(Buffer(arbitrary))
        if (data.dtype in
           [np.float16, np.float32, np.float64] and arbitrary.size > 0):
            if nan_as_null:
                mask = cudautils.mask_from_devary(arbitrary)
                data = data.set_mask(mask)

    elif cuda.is_cuda_array(arbitrary):
        # Use cuda array interface to do create a numba device array by
        # reference
        new_dev_array = cuda.as_cuda_array(arbitrary)

        # Allocate new output array using rmm and copy the numba device array
        # to an rmm owned device array
        out_dev_array = rmm.device_array_like(new_dev_array)
        out_dev_array.copy_to_device(new_dev_array)

        data = as_column(out_dev_array)

    elif isinstance(arbitrary, np.ndarray):
        # CUDF assumes values are always contiguous
        if not arbitrary.flags['C_CONTIGUOUS']:
            arbitrary = np.ascontiguousarray(arbitrary)
        if arbitrary.dtype.kind == 'M':
            data = datetime.DatetimeColumn.from_numpy(arbitrary)
        elif arbitrary.dtype.kind in ('O', 'U'):
            data = as_column(pa.Array.from_pandas(arbitrary))
        else:
            data = as_column(rmm.to_device(arbitrary), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, pa.Array):
        if isinstance(arbitrary, pa.StringArray):
            count = len(arbitrary)
            null_count = arbitrary.null_count

            buffers = arbitrary.buffers()
            # Buffer of actual strings values
            if buffers[2] is not None:
                sbuf = np.frombuffer(buffers[2], dtype='int8')
            else:
                sbuf = np.empty(0, dtype='int8')
            # Buffer of offsets values
            obuf = np.frombuffer(buffers[1], dtype='int32')
            # Buffer of null bitmask
            nbuf = None
            if null_count > 0:
                nbuf = np.frombuffer(buffers[0], dtype='int8')

            data = as_column(
                nvstrings.from_offsets(sbuf, obuf, count, nbuf=nbuf,
                                       ncount=null_count)
            )
        elif isinstance(arbitrary, pa.NullArray):
            new_dtype = dtype
            if (type(dtype) == str and dtype == 'empty') or dtype is None:
                new_dtype = np.dtype(arbitrary.type.to_pandas_dtype())

            if pd.api.types.is_categorical_dtype(new_dtype):
                arbitrary = arbitrary.dictionary_encode()
            else:
                if nan_as_null:
                    arbitrary = arbitrary.cast(np_to_pa_dtype(new_dtype))
                else:
                    # casting a null array doesn't make nans valid
                    # so we create one with valid nans from scratch:
                    if new_dtype == np.dtype("object"):
                        arbitrary = utils.scalar_broadcast_to(
                            None,
                            (len(arbitrary),),
                            dtype=new_dtype
                        )
                    else:
                        arbitrary = utils.scalar_broadcast_to(
                            np.nan,
                            (len(arbitrary),),
                            dtype=new_dtype
                        )
            data = as_column(arbitrary, nan_as_null=nan_as_null)
        elif isinstance(arbitrary, pa.DictionaryArray):
            pamask, padata = buffers_from_pyarrow(arbitrary)
            data = categorical.CategoricalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                categories=arbitrary.dictionary.to_pylist(),
                ordered=arbitrary.type.ordered,
            )
        elif isinstance(arbitrary, pa.TimestampArray):
            arbitrary = arbitrary.cast(pa.timestamp('ms'))
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype='M8[ms]')
            data = datetime.DatetimeColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype('M8[ms]')
            )
        elif isinstance(arbitrary, pa.Date64Array):
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype='M8[ms]')
            data = datetime.DatetimeColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype('M8[ms]')
            )
        elif isinstance(arbitrary, pa.Date32Array):
            # No equivalent np dtype and not yet supported
            warnings.warn("Date32 values are not yet supported so this will "
                          "be typecast to a Date64 value", UserWarning)
            arbitrary = arbitrary.cast(pa.date64())
            data = as_column(arbitrary)
        elif isinstance(arbitrary, pa.BooleanArray):
            # Arrow uses 1 bit per value while we use int8
            dtype = np.dtype(np.bool)
            # Needed because of bug in PyArrow
            # https://issues.apache.org/jira/browse/ARROW-4766
            if len(arbitrary) > 0:
                arbitrary = arbitrary.cast(pa.int8())
            else:
                arbitrary = pa.array([], type=pa.int8())
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype=dtype)
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=dtype
            )
        else:
            pamask, padata = buffers_from_pyarrow(arbitrary)
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype(arbitrary.type.to_pandas_dtype())
            )

    elif isinstance(arbitrary, pa.ChunkedArray):
        gpu_cols = [as_column(chunk, dtype=dtype) for chunk in
                    arbitrary.chunks]

        if dtype and dtype != 'empty':
            new_dtype = dtype
        else:
            pa_type = arbitrary.type
            if pa.types.is_dictionary(pa_type):
                new_dtype = 'category'
            else:
                new_dtype = np.dtype(pa_type.to_pandas_dtype())

        data = Column._concat(gpu_cols, dtype=new_dtype)

    elif isinstance(arbitrary, (pd.Series, pd.Categorical)):
        if pd.api.types.is_categorical_dtype(arbitrary):
            data = as_column(pa.array(arbitrary, from_pandas=True))
        elif arbitrary.dtype == np.bool:
            # Bug in PyArrow or HDF that requires us to do this
            data = as_column(pa.array(np.array(arbitrary), from_pandas=True))
        else:
            data = as_column(pa.array(arbitrary, from_pandas=nan_as_null))

    elif isinstance(arbitrary, pd.Timestamp):
        arbitrary = arbitrary.ceil('ms')
        # This will always treat NaTs as nulls since it's not technically a
        # discrete value like NaN
        data = as_column(pa.array(pd.Series([arbitrary]), from_pandas=True))

    elif np.isscalar(arbitrary) and not isinstance(arbitrary, memoryview):
        if hasattr(arbitrary, 'dtype'):
            data_type = np_to_pa_dtype(arbitrary.dtype)
            if data_type in (pa.date64(), pa.date32()):
                # PyArrow can't construct date64 or date32 arrays from np
                # datetime types
                arbitrary = arbitrary.astype('int64')
            data = as_column(pa.array([arbitrary], type=data_type))
        else:
            data = as_column(pa.array([arbitrary]), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, memoryview):
        data = as_column(np.array(arbitrary), dtype=dtype,
                         nan_as_null=nan_as_null)

    else:
        try:
            data = as_column(memoryview(arbitrary))
        except TypeError:
            try:
                pa_type = None
                if dtype is not None:
                    if pd.api.types.is_categorical_dtype(dtype):
                        raise TypeError
                    else:
                        np_type = np.dtype(dtype).type
                        if np_type == np.bool_:
                            pa_type = pa.bool_()
                        else:
                            pa_type = np_to_pa_dtype(np.dtype(dtype).type)
                data = as_column(
                    pa.array(arbitrary, type=pa_type, from_pandas=nan_as_null),
                    nan_as_null=nan_as_null
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                np_type = None
                if pd.api.types.is_categorical_dtype(dtype):
                    data = as_column(
                        pd.Series(arbitrary, dtype='category'),
                        nan_as_null=nan_as_null
                    )
                else:
                    if dtype is None:
                        np_type = None
                    else:
                        np_type = np.dtype(dtype)
                    data = as_column(
                        np.array(arbitrary, dtype=np_type),
                        nan_as_null=nan_as_null
                    )

    return data


def column_applymap(udf, column, out_dtype):
    """Apply a elemenwise function to transform the values in the Column.

    Parameters
    ----------
    udf : function
        Wrapped by numba jit for call on the GPU as a device function.
    column : Column
        The source column.
    out_dtype  : numpy.dtype
        The dtype for use in the output.

    Returns
    -------
    result : Buffer
    """
    core = njit(udf)
    results = rmm.device_array(shape=len(column), dtype=out_dtype)
    values = column.data.to_gpu_array()
    if column.mask:
        # For masked columns
        @cuda.jit
        def kernel_masked(values, masks, results):
            i = cuda.grid(1)
            # in range?
            if i < values.size:
                # valid?
                if utils.mask_get(masks, i):
                    # call udf
                    results[i] = core(values[i])

        masks = column.mask.to_gpu_array()
        kernel_masked.forall(len(column))(values, masks, results)
    else:
        # For non-masked columns
        @cuda.jit
        def kernel_non_masked(values, results):
            i = cuda.grid(1)
            # in range?
            if i < values.size:
                # call udf
                results[i] = core(values[i])

        kernel_non_masked.forall(len(column))(values, results)
    # Output
    return Buffer(results)
