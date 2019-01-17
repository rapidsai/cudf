"""
Provides base classes and utils for implementing type-specific logical
view of Columns.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from numba import cuda, njit

from librmm_cffi import librmm as rmm

from .buffer import Buffer
from .column import Column
from cudf.utils import utils, cudautils
from cudf import _gdf

import warnings


class TypedColumnBase(Column):
    """Base class for all typed column
    e.g. NumericalColumn, CategoricalColumn

    This class provides common operations to implement logical view and
    type-based operations for the column.

    Notes
    -----
    For designed to be instantiated directly.  Instantiate subclasses instead.
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

    def argsort(self, ascending):
        _, inds = self.sort_by_values(ascending=ascending)
        return inds

    def sort_by_values(self, ascending):
        raise NotImplementedError


def column_empty_like(column, dtype, masked):
    """Allocate a new column like the given *column*
    """
    data = rmm.device_array(shape=len(column), dtype=dtype)
    params = dict(data=Buffer(data))
    if masked:
        mask = utils.make_mask(data.size)
        params.update(dict(mask=Buffer(mask), null_count=data.size))
    return Column(**params)


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
    from .numerical import NumericalColumn
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
    from .numerical import NumericalColumn
    assert column.null_count == 0

    selvals = cudautils.gather(column.data.to_gpu_array(),
                               positions.data.to_gpu_array())

    selected_values = column.replace(data=Buffer(selvals))
    selected_index = Buffer(positions.data.to_gpu_array())

    return selected_values, NumericalColumn(data=selected_index,
                                            dtype=selected_index.dtype)


def as_column(arbitrary, nan_as_null=True):
    """Create a Column from an arbitrary object

    Currently support inputs are:

    * ``Column``
    * ``Buffer``
    * numba device array
    * numpy array
    * pandas.Categorical

    Returns
    -------
    result : subclass of TypedColumnBase
        - CategoricalColumn for pandas.Categorical input.
        - NumericalColumn for all other inputs.
    """
    from . import numerical, categorical, datetime

    if isinstance(arbitrary, Column):
        if not isinstance(arbitrary, TypedColumnBase):
            # interpret as numeric
            data = arbitrary.view(numerical.NumericalColumn,
                                  dtype=arbitrary.dtype)
        else:
            data = arbitrary

    elif isinstance(arbitrary, Buffer):
        data = numerical.NumericalColumn(data=arbitrary, dtype=arbitrary.dtype)

    elif cuda.devicearray.is_cuda_ndarray(arbitrary):
        data = as_column(Buffer(arbitrary))
        if (data.dtype in [np.float16, np.float32, np.float64]
                and arbitrary.size > 0):
            if nan_as_null:
                mask = cudautils.mask_from_devary(arbitrary)
                data = data.set_mask(mask)

    elif isinstance(arbitrary, np.ndarray):
        if arbitrary.dtype.kind == 'M':
            data = datetime.DatetimeColumn.from_numpy(arbitrary)
        elif arbitrary.dtype.kind in ('O', 'U'):
            raise NotImplementedError("Strings are not yet supported")
        else:
            data = as_column(rmm.to_device(arbitrary), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, pa.Array):
        if isinstance(arbitrary, pa.StringArray):
            raise NotImplementedError("Strings are not yet supported")
        elif isinstance(arbitrary, pa.NullArray):
            pamask = Buffer(np.empty(0, dtype='int8'))
            padata = Buffer(
                np.empty(0, dtype=arbitrary.type.to_pandas_dtype())
            )
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=0,
                dtype=np.dtype(arbitrary.type.to_pandas_dtype())
            )
        elif isinstance(arbitrary, pa.DictionaryArray):
            if arbitrary.buffers()[0]:
                pamask = Buffer(np.array(arbitrary.buffers()[0]))
            else:
                pamask = None
            padata = Buffer(np.array(arbitrary.buffers()[1]).view(
                arbitrary.indices.type.to_pandas_dtype()
            ))
            data = categorical.CategoricalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                categories=arbitrary.dictionary.to_pylist(),
                ordered=arbitrary.type.ordered,
            )
        elif isinstance(arbitrary, pa.TimestampArray):
            arbitrary = arbitrary.cast(pa.timestamp('ms'))
            if arbitrary.buffers()[0]:
                pamask = Buffer(np.array(arbitrary.buffers()[0]))
            else:
                pamask = None
            padata = Buffer(np.array(arbitrary.buffers()[1]).view(
                np.dtype('M8[ms]')
            ))
            data = datetime.DatetimeColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype('M8[ms]')
            )
        elif isinstance(arbitrary, pa.Date64Array):
            if arbitrary.buffers()[0]:
                pamask = Buffer(np.array(arbitrary.buffers()[0]))
            else:
                pamask = None
            padata = Buffer(np.array(arbitrary.buffers()[1]).view(
                np.dtype('M8[ms]')
            ))
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
            arbitrary = arbitrary.cast(pa.int8())
            if arbitrary.buffers()[0]:
                pamask = Buffer(np.array(arbitrary.buffers()[0]))
            else:
                pamask = None
            padata = Buffer(np.array(arbitrary.buffers()[1]).view(
                dtype
            ))
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=dtype
            )
        else:
            if arbitrary.buffers()[0]:
                pamask = Buffer(np.array(arbitrary.buffers()[0]))
            else:
                pamask = None
            padata = Buffer(np.array(arbitrary.buffers()[1]).view(
                np.dtype(arbitrary.type.to_pandas_dtype())
            ))
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype(arbitrary.type.to_pandas_dtype())
            )

    elif isinstance(arbitrary, (pd.Series, pd.Categorical)):
        if pd.api.types.is_categorical_dtype(arbitrary):
            data = as_column(pa.array(arbitrary, from_pandas=True))
        else:
            data = as_column(pa.array(arbitrary, from_pandas=nan_as_null))

    elif np.isscalar(arbitrary) and not isinstance(arbitrary, memoryview):
        if hasattr(arbitrary, 'dtype'):
            data_type = _gdf.np_to_pa_dtype(arbitrary.dtype)
            if data_type in (pa.date64(), pa.date32()):
                # PyArrow can't construct date64 or date32 arrays from np
                # datetime types
                arbitrary = arbitrary.astype('int64')
            data = as_column(pa.array([arbitrary], type=data_type))
        else:
            data = as_column(pa.array([arbitrary]))

    elif isinstance(arbitrary, memoryview):
        data = as_column(np.array(arbitrary))

    else:
        try:
            data = as_column(memoryview(arbitrary))
        except TypeError:
            data = as_column(pa.array(arbitrary))

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
