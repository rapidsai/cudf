# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Sequence, Union
#
# from cudf._typing import Dtype, DtypeObj, ScalarLike
# from cudf.core.buffer import Buffer
# from cudf.core.column import ColumnBase
#
# T = TypeVar("T")


class Reducible:
    def sum(self, *args, **kwargs):
        ...

    def product(self, *args, **kwargs):
        ...

    def min(self, *args, **kwargs):
        ...

    def max(self, *args, **kwargs):
        ...

    def count(self, *args, **kwargs):
        ...

    def size(self, *args, **kwargs):
        ...

    def any(self, *args, **kwargs):
        ...

    def all(self, *args, **kwargs):
        ...

    def sum_of_squares(self, *args, **kwargs):
        ...

    def mean(self, *args, **kwargs):
        ...

    def var(self, *args, **kwargs):
        ...

    def std(self, *args, **kwargs):
        ...

    def median(self, *args, **kwargs):
        ...

    def quantile(self, q: Union[float, Sequence[float]], interpolation: str, exact: bool, *args, **kwargs):
        ...

    def argmax(self, *args, **kwargs):
        ...

    def argmin(self, *args, **kwargs):
        ...

    def nunique(self, *args, **kwargs):
        ...

    def nth(self, *args, **kwargs):
        ...

    def collect(self, *args, **kwargs):
        ...

    def unique(self, *args, **kwargs):
        ...

    def prod(self, *args, **kwargs):
        ...

    def idxmin(self, *args, **kwargs):
        ...

    def idxmax(self, *args, **kwargs):
        ...

    def first(self, *args, **kwargs):
        ...

    def last(self, *args, **kwargs):
        ...
