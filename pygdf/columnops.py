import numpy as np
import pandas as pd

from numba import cuda

from .buffer import Buffer
from . import utils, cudautils
from .column import Column


class ColumnOps(Column):
    def __init__(self, **kwargs):
        dtype = kwargs.pop('dtype')
        super(ColumnOps, self).__init__(**kwargs)
        # Logical dtype
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def is_type_equivalent(self, other):
        mine = self._replace_defaults()
        theirs = other._replace_defaults()

        def remove_base(dct):
            basekeys = Column._replace_defaults(self).keys()
            for k in basekeys:
                del dct[k]

        remove_base(mine)
        remove_base(theirs)

        return type(self) == type(other) and mine == theirs

    def _replace_defaults(self):
        params = super(ColumnOps, self)._replace_defaults()
        params.update(dict(dtype=self._dtype))
        return params

    def argsort(self, ascending):
        _, inds = self.sort_by_values(ascending=ascending)
        return inds


def column_empty_like(column, dtype, masked):
    """Allocate a new column like the given *column*
    """
    data = cuda.device_array(shape=len(column), dtype=dtype)
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
    data = cuda.device_array(shape=len(column), dtype=dtype)
    params = dict(data=Buffer(data))
    if column.has_null_mask:
        params.update(mask=column.nullmask)
    return Column(**params)


def column_select_by_boolmask(column, boolmask):
    """Select by a boolean mask to a column.

    Returns (selected_column, selected_positions)
    """
    from .numerical import NumericalColumn
    assert not column.has_null_mask
    boolbits = cudautils.compact_mask_bytes(boolmask.to_gpu_array())
    indices = cudautils.arange(len(boolmask))
    _, selinds = cudautils.copy_to_dense(indices, mask=boolbits)
    _, selvals = cudautils.copy_to_dense(column.data.to_gpu_array(),
                                         mask=boolbits)

    assert not column.has_null_mask   # the nullmask needs to be recomputed

    selected_values = column.replace(data=Buffer(selvals))
    selected_index = Buffer(selinds)
    return selected_values, NumericalColumn(data=selected_index,
                                            dtype=selected_index.dtype)


def as_column(arbitrary):
    """Create a Column from an arbitrary object

    Currently support inputs are:

    * ``Column``
    * ``Buffer``
    * numba device array
    * numpy array
    * pandas.Categorical
    """
    from . import numerical, categorical

    if isinstance(arbitrary, Column):
        if not isinstance(arbitrary, ColumnOps):
            # interpret as numeric
            return arbitrary.view(numerical.NumericalColumn,
                                  dtype=arbitrary.dtype)
        else:
            return arbitrary
    elif isinstance(arbitrary, pd.Categorical):
        return categorical.pandas_categorical_as_column(arbitrary)
    elif isinstance(arbitrary, Buffer):
        return numerical.NumericalColumn(data=arbitrary, dtype=arbitrary.dtype)
    elif cuda.devicearray.is_cuda_ndarray(arbitrary):
        return as_column(Buffer(arbitrary))
    elif isinstance(arbitrary, np.ndarray):
        return as_column(Buffer(arbitrary))
    else:
        return as_column(np.asarray(arbitrary))

