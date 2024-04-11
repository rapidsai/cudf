# Copyright (c) 2024, NVIDIA CORPORATION.

from functools import cached_property

from dask_expr import (
    DataFrame as DXDataFrame,
    FrameBase,
    Index as DXIndex,
    Series as DXSeries,
    get_collection_type,
)
from dask_expr._collection import new_collection
from dask_expr._util import _raise_if_object_series

from dask import config
from dask.dataframe.core import is_dataframe_like

import cudf

##
## Custom collection classes
##


# VarMixin can be removed if cudf#15179 is addressed.
# See: https://github.com/rapidsai/cudf/issues/15179
class VarMixin:
    def var(
        self,
        axis=0,
        skipna=True,
        ddof=1,
        numeric_only=False,
        split_every=False,
        **kwargs,
    ):
        _raise_if_object_series(self, "var")
        axis = self._validate_axis(axis)
        self._meta.var(axis=axis, skipna=skipna, numeric_only=numeric_only)
        frame = self
        if is_dataframe_like(self._meta) and numeric_only:
            # Convert to pandas - cudf does something weird here
            index = self._meta.to_pandas().var(numeric_only=True).index
            frame = frame[list(index)]
        return new_collection(
            frame.expr.var(
                axis, skipna, ddof, numeric_only, split_every=split_every
            )
        )


class DataFrame(VarMixin, DXDataFrame):
    @classmethod
    def from_dict(cls, *args, **kwargs):
        with config.set({"dataframe.backend": "cudf"}):
            return DXDataFrame.from_dict(*args, **kwargs)

    def groupby(
        self,
        by,
        group_keys=True,
        sort=None,
        observed=None,
        dropna=None,
        **kwargs,
    ):
        from dask_cudf.expr._groupby import GroupBy

        if isinstance(by, FrameBase) and not isinstance(by, DXSeries):
            raise ValueError(
                f"`by` must be a column name or list of columns, got {by}."
            )

        return GroupBy(
            self,
            by,
            group_keys=group_keys,
            sort=sort,
            observed=observed,
            dropna=dropna,
            **kwargs,
        )

    def to_orc(self, *args, **kwargs):
        return self.to_legacy_dataframe().to_orc(*args, **kwargs)

    @staticmethod
    def read_text(*args, **kwargs):
        from dask_expr import from_legacy_dataframe

        from dask_cudf.io.text import read_text as legacy_read_text

        ddf = legacy_read_text(*args, **kwargs)
        return from_legacy_dataframe(ddf)


class Series(VarMixin, DXSeries):
    def groupby(self, by, **kwargs):
        from dask_cudf.expr._groupby import SeriesGroupBy

        return SeriesGroupBy(self, by, **kwargs)

    @cached_property
    def list(self):
        from dask_cudf.accessors import ListMethods

        return ListMethods(self)

    @cached_property
    def struct(self):
        from dask_cudf.accessors import StructMethods

        return StructMethods(self)


class Index(DXIndex):
    pass  # Same as pandas (for now)


get_collection_type.register(cudf.DataFrame, lambda _: DataFrame)
get_collection_type.register(cudf.Series, lambda _: Series)
get_collection_type.register(cudf.BaseIndex, lambda _: Index)


##
## Support conversion to GPU-backed Array collections
##


try:
    from dask_expr._backends import create_array_collection

    @get_collection_type.register_lazy("cupy")
    def _register_cupy():
        import cupy

        @get_collection_type.register(cupy.ndarray)
        def get_collection_type_cupy_array(_):
            return create_array_collection

    @get_collection_type.register_lazy("cupyx")
    def _register_cupyx():
        # Needed for cuml
        from cupyx.scipy.sparse import spmatrix

        @get_collection_type.register(spmatrix)
        def get_collection_type_csr_matrix(_):
            return create_array_collection

except ImportError:
    # Older version of dask-expr.
    # Implicit conversion to array wont work.
    pass
