class SeriesImpl(object):
    """
    Provides type-based delegation of operations on a Series.

    The ``Series`` class delegate the implementation of each operations
    to the a subclass of ``SeriesImpl``.  Depending of the dtype of the
    Series, it will load the corresponding implementation of the
    ``SeriesImpl``.
    """
    def __init__(self, dtype):
        self._dtype = dtype

    def __eq__(self, other):
        return self.dtype == other.dtype

    def __ne__(self, other):
        out = self.dtype == other.dtype
        if out is NotImplemented:
            return out
        return not out

    @property
    def dtype(self):
        return self._dtype

    # Methods below are all overridable

    def cat(self, series):
        raise TypeError('not a categorical series')

    def element_to_str(self, value):
        raise NotImplementedError

    def binary_operator(self, binop, lhs, rhs):
        raise NotImplementedError

    def unary_operator(self, unaryop, series):
        raise NotImplementedError

    def unordered_compare(self, cmpop, lhs, rhs):
        raise NotImplementedError

    def ordered_compare(self, cmpop, lhs, rhs):
        raise NotImplementedError
