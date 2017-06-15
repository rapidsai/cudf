from .dataframe import Series
from .series_impl import SeriesImpl


class CategoricalAccessor(object):
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
    def __init__(self, categories, ordered):
        self._categories = categories
        self._ordered = ordered

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
        return str(self._decode(value))
