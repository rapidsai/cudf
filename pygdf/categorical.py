import pandas as pd
import numpy as np

from .dataframe import Series
from . import numerical, utils, columnops
from .buffer import Buffer
from . import cudautils


class CategoricalAccessor(object):
    """
    This mimicks pandas `df.cat` interface.
    """
    def __init__(self, parent, categories, ordered):
        self._parent = parent
        self._categories = tuple(categories)
        self._ordered = ordered

    @property
    def categories(self):
        return self._categories

    @property
    def ordered(self):
        return self._ordered

    @property
    def codes(self):
        data = self._parent.data
        if self._parent.has_null_mask:
            mask = self._parent.mask
            null_count = self._parent.null_count
            return Series.from_masked_array(data=data.mem, mask=mask.mem,
                                            null_count=null_count)
        else:
            return Series(data)


class CategoricalColumn(columnops.TypedColumnBase):
    """Implements operations for Columns of Categorical type
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        data : Buffer
            The code values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        categories : iterable
            The categories
        ordered : bool
            whether the categorical has a logical ordering (e.g. less than)
        """
        categories = kwargs.pop('categories')
        ordered = kwargs.pop('ordered')
        super(CategoricalColumn, self).__init__(**kwargs)
        self._categories = tuple(categories)
        self._ordered = bool(ordered)

    def _replace_defaults(self):
        params = super(CategoricalColumn, self)._replace_defaults()
        params.update(dict(categories=self._categories,
                           ordered=self._ordered))
        return params

    @property
    def as_numerical(self):
        return self.view(numerical.NumericalColumn, dtype=self.data.dtype)

    def cat(self):
        return CategoricalAccessor(self, categories=self._categories,
                                   ordered=self._ordered)

    def binary_operator(self, binop, rhs):
        msg = 'Categorical cannot perform the operation: {}'.format(binop)
        raise TypeError(msg)

    def unary_operator(self, unaryop):
        msg = 'Categorical cannot perform the operation: {}'.format(unaryop)
        raise TypeError(msg)

    def unordered_compare(self, cmpop, rhs):
        if not self.is_type_equivalent(rhs):
            raise TypeError('Categoricals can only compare with the same type')
        return self.as_numerical.unordered_compare(cmpop, rhs.as_numerical)

    def ordered_compare(self, cmpop, rhs):
        if not (self._ordered and rhs._ordered):
            msg = "Unordered Categoricals can only compare equality or not"
            raise TypeError(msg)
        if not self.is_type_equivalent(rhs):
            raise TypeError('Categoricals can only compare with the same type')
        return self.as_numerical.ordered_compare(cmpop, rhs.as_numerical)

    def normalize_binop_value(self, other):
        ary = utils.scalar_broadcast_to(self._encode(other), shape=len(self),
                                        dtype=self.data.dtype)
        col = self.replace(data=Buffer(ary), dtype=self.dtype,
                           categories=self._categories, ordered=self._ordered)
        return col

    def astype(self, dtype):
        # custom dtype can't be compared with `==`
        if self.dtype is dtype:
            return self
        return self.as_numerical.astype(dtype)

    def sort_by_values(self, ascending):
        return self.as_numerical.sort_by_values(ascending)

    def element_indexing(self, index):
        val = self.as_numerical.element_indexing(index)
        return self._decode(val) if val is not None else val

    def to_pandas(self, index=None):
        data = pd.Categorical.from_codes(self.cat().codes.to_array(),
                                         categories=self._categories,
                                         ordered=self._ordered)
        return pd.Series(data, index=index)

    def _encode(self, value):
        for i, cat in enumerate(self._categories):
            if cat == value:
                return i
        return -1

    def _decode(self, value):
        for i, cat in enumerate(self._categories):
            if i == value:
                return cat

    def default_na_value(self):
        return -1


def pandas_categorical_as_column(categorical, codes=None):
    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    # TODO fix mutability issue in numba to avoid the .copy()
    codes = (categorical.codes.copy() if codes is None else codes)
    # TODO pending pandas to be improved
    #       https://github.com/pandas-dev/pandas/issues/14711
    #       https://github.com/pandas-dev/pandas/pull/16015
    valid_codes = codes != -1
    buf = Buffer(codes)
    params = dict(data=buf, dtype=categorical.dtype,
                  categories=categorical.categories,
                  ordered=categorical.ordered)
    if not np.all(valid_codes):
        mask = cudautils.compact_mask_bytes(valid_codes)
        nnz = np.count_nonzero(valid_codes)
        null_count = codes.size - nnz
        params.update(dict(mask=Buffer(mask), null_count=null_count))

    return CategoricalColumn(**params)
