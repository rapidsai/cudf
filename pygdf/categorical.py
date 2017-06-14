from .dataframe import Series


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

    def _encode(self, value):
        for i, cat in enumerate(self.categories):
            if cat == value:
                return i
        return -1

    def _decode(self, value):
        for i, cat in enumerate(self.categories):
            if i == value:
                return cat


class CategoricalSeries(Series):
    def __init__(self, *args, **kwargs):
        categories = kwargs.pop('categories')
        ordered = kwargs.pop('ordered')
        self._cat = CategoricalAccessor(self, categories=categories,
                                        ordered=ordered)
        super(CategoricalSeries, self).__init__(*args, **kwargs)

    @property
    def cat(self):
        return self._cat

    def _element_to_str(self, value):
        """Overriding
        """
        return str(self.cat._decode(value))

    def _copy_construct_defaults(self):
        """Overriding
        """
        params = super(CategoricalSeries, self)._copy_construct_defaults()
        params['categories'] = self._cat.categories
        params['ordered'] = self._cat.ordered
        return params
