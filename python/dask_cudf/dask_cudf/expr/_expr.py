# Copyright (c) 2024, NVIDIA CORPORATION.
import functools

import numpy as np
from dask_expr._cumulative import CumulativeBlockwise
from dask_expr._expr import Expr, VarColumns
from dask_expr._quantiles import RepartitionQuantiles
from dask_expr._reductions import Reduction, Var

from dask.dataframe.core import is_dataframe_like, make_meta, meta_nonempty

##
## Custom expression patching
##


# This can be removed after cudf#15176 is addressed.
# See: https://github.com/rapidsai/cudf/issues/15176
class PatchCumulativeBlockwise(CumulativeBlockwise):
    @property
    def _args(self) -> list:
        return self.operands[:1]

    @property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


CumulativeBlockwise._args = PatchCumulativeBlockwise._args
CumulativeBlockwise._kwargs = PatchCumulativeBlockwise._kwargs


# The upstream Var code uses `Series.values`, and relies on numpy
# for most of the logic. Unfortunately, cudf -> cupy conversion
# is not supported for data containing null values. Therefore,
# we must implement our own version of Var for now. This logic
# is mostly copied from dask-cudf.


class VarCudf(Reduction):
    # Uses the parallel version of Welford's online algorithm (Chan '79)
    # (http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
    _parameters = ["frame", "skipna", "ddof", "numeric_only", "split_every"]
    _defaults = {
        "skipna": True,
        "ddof": 1,
        "numeric_only": False,
        "split_every": False,
    }

    @functools.cached_property
    def _meta(self):
        return make_meta(
            meta_nonempty(self.frame._meta).var(
                skipna=self.skipna, numeric_only=self.numeric_only
            )
        )

    @property
    def chunk_kwargs(self):
        return dict(skipna=self.skipna, numeric_only=self.numeric_only)

    @property
    def combine_kwargs(self):
        return {}

    @property
    def aggregate_kwargs(self):
        return dict(ddof=self.ddof)

    @classmethod
    def reduction_chunk(cls, x, skipna=True, numeric_only=False):
        kwargs = {"numeric_only": numeric_only} if is_dataframe_like(x) else {}
        if skipna or numeric_only:
            n = x.count(**kwargs)
            kwargs["skipna"] = skipna
            avg = x.mean(**kwargs)
        else:
            # Not skipping nulls, so might as well
            # avoid the full `count` operation
            n = len(x)
            kwargs["skipna"] = skipna
            avg = x.sum(**kwargs) / n
        if numeric_only:
            # Workaround for cudf bug
            # (see: https://github.com/rapidsai/cudf/issues/13731)
            x = x[n.index]
        m2 = ((x - avg) ** 2).sum(**kwargs)
        return n, avg, m2

    @classmethod
    def reduction_combine(cls, parts):
        n, avg, m2 = parts[0]
        for i in range(1, len(parts)):
            n_a, avg_a, m2_a = n, avg, m2
            n_b, avg_b, m2_b = parts[i]
            n = n_a + n_b
            avg = (n_a * avg_a + n_b * avg_b) / n
            delta = avg_b - avg_a
            m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
        return n, avg, m2

    @classmethod
    def reduction_aggregate(cls, vals, ddof=1):
        vals = cls.reduction_combine(vals)
        n, _, m2 = vals
        return m2 / (n - ddof)


def _patched_var(
    self, axis=0, skipna=True, ddof=1, numeric_only=False, split_every=False
):
    if axis == 0:
        if hasattr(self._meta, "to_pandas"):
            return VarCudf(self, skipna, ddof, numeric_only, split_every)
        else:
            return Var(self, skipna, ddof, numeric_only, split_every)
    elif axis == 1:
        return VarColumns(self, skipna, ddof, numeric_only)
    else:
        raise ValueError(f"axis={axis} not supported. Please specify 0 or 1")


Expr.var = _patched_var


# Add custom code path for RepartitionQuantiles, because
# upstream logic fails when null values are present. Note
# that the cudf-specific code path can also be used for
# multi-column divisions in the future.


def _quantile(a, q):
    if a.empty:
        # Avoid calling `quantile` on empty data
        return None, 0
    a = a.to_frame() if a.ndim == 1 else a
    return (
        a.quantile(q=q.tolist(), interpolation="nearest", method="table"),
        len(a),
    )


def merge_quantiles(finalq, qs, vals):
    from dask_cudf.sorting import merge_quantiles as mq

    return mq(finalq, qs, vals).iloc[:, 0].to_pandas()


_original_layer = RepartitionQuantiles._layer


def _cudf_layer(self):
    if hasattr(self._meta, "to_pandas"):
        # pandas/cudf uses quantile in [0, 1]
        # numpy / cupy uses [0, 100]
        qs = np.linspace(0.0, 1.0, self.input_npartitions + 1)
        val_dsk = {
            (self._name, 0, i): (_quantile, key, qs)
            for i, key in enumerate(self.frame.__dask_keys__())
        }
        merge_dsk = {
            (self._name, 0): (
                merge_quantiles,
                qs,
                [qs] * self.input_npartitions,
                sorted(val_dsk),
            )
        }
        return {**val_dsk, **merge_dsk}
    else:
        return _original_layer(self)


RepartitionQuantiles._layer = _cudf_layer
