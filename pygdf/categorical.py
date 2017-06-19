from .dataframe import Series
from .series_impl import SeriesImpl
from . import numerical


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
            mask = self._parent._mask
            null_count = self._parent.null_count
            return Series.from_masked_array(data=data.mem, mask=mask.mem,
                                            null_count=null_count)
        else:
            return Series.from_buffer(data)


class CategoricalSeriesImpl(SeriesImpl):
    """
    Implements a Categorical Series that treats integral values as index
    into a dictionary that map to arbitrary objects (e.g. string).
    """
    def __init__(self, dtype, codes_dtype, categories, ordered):
        super(CategoricalSeriesImpl, self).__init__(dtype)
        self._categories = categories
        self._ordered = ordered
        # This contains the `.code` series implementation
        self._codes_impl = numerical.NumericalSeriesImpl(codes_dtype)

    def __eq__(self, other):
        if isinstance(other, CategoricalSeriesImpl):
            return all([self.dtype == other.dtype,
                        tuple(self._categories) == tuple(other._categories),
                        self._ordered == other._ordered,
                        self._codes_impl == other._codes_impl])

    def _encode(self, value):
        for i, cat in enumerate(self._categories):
            if cat == value:
                return i
        return -1

    def _decode(self, value):
        for i, cat in enumerate(self._categories):
            if i == value:
                return cat

    def cat(self, series):
        return CategoricalAccessor(series, categories=self._categories,
                                   ordered=self._ordered)

    def element_to_str(self, value):
        return str(value)

    def binary_operator(self, binop, lhs, rhs):
        msg = 'Categorical cannot perform the operation: {}'.format(binop)
        raise TypeError(msg)

    def unary_operator(self, unaryop, series):
        msg = 'Categorical cannot perform the operation: {}'.format(unaryop)
        raise TypeError(msg)

    def unordered_compare(self, cmpop, lhs, rhs):
        if self != rhs._impl:
            raise TypeError('Categoricals can only compare with the same type')
        return self._codes_impl.unordered_compare(cmpop, lhs, rhs)

    def ordered_compare(self, cmpop, lhs, rhs):
        if not (self._ordered and rhs._impl._ordered):
            msg = "Unordered Categoricals can only compare equality or not"
            raise TypeError(msg)
        if self != rhs._impl:
            raise TypeError('Categoricals can only compare with the same type')
        return self._codes_impl.ordered_compare(cmpop, lhs, rhs)

    def element_indexing(self, series, index):
        val = self._codes_impl.element_indexing(series, index)
        return self._decode(val) if val is not None else val
