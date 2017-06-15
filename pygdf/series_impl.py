class SeriesImpl(object):
    """
    Provides type-based delegation of operations on a Series.
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

    def cat(self, series):
        raise TypeError('not a categorical series')

