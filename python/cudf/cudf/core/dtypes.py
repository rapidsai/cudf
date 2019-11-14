import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype

import cudf


class CategoricalDtype(ExtensionDtype):
    def __init__(self, data_dtype, categories=None, ordered=None):
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self.data_dtype = np.dtype(data_dtype)
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    @property
    def categories(self):
        if self._categories is None:
            return None
        return cudf.core.index.as_index(self._categories)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    def to_pandas(self):
        if self.categories is None:
            categories = None
        else:
            categories = self.categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories):
        if categories is None:
            return categories
        if len(categories) == 0:
            dtype = "object"
        else:
            dtype = None
        return cudf.core.column.as_column(categories, dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        elif self.categories is None or other.categories is None:
            return True
        else:
            return (
                self.data_dtype == other.data_dtype
                and self._categories.dtype == other._categories.dtype
                and self._categories.equals(other._categories)
            )

    def construct_from_string(self):
        raise NotImplementedError()

    def serialize(self):
        header = {}
        frames = []
        header["data_dtype"] = self.data_dtype.str
        header["ordered"] = self.ordered
        categories_buffer = []
        if self.categories is not None:
            categories_buffer = [self.categories.as_column()._data_view()]
        frames.extend(categories_buffer)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        data_dtype = header["data_dtype"]
        ordered = header["ordered"]
        categories = frames[0]
        return cls(
            data_dtype=data_dtype, categories=categories, ordered=ordered
        )
