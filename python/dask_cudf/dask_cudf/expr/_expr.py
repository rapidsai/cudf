# Copyright (c) 2024, NVIDIA CORPORATION.
import functools

import dask_expr._shuffle as _shuffle_module
import pandas as pd
from dask_expr import new_collection
from dask_expr._cumulative import CumulativeBlockwise
from dask_expr._expr import Elemwise, Expr, VarColumns
from dask_expr._reductions import Reduction, Var
from dask_expr.io.io import FusedParquetIO
from dask_expr.io.parquet import ReadParquetPyarrowFS

from dask.dataframe.core import is_dataframe_like, make_meta, meta_nonempty
from dask.dataframe.dispatch import is_categorical_dtype

import cudf

##
## Custom expressions
##


class CudfFusedParquetIO(FusedParquetIO):
    @staticmethod
    def _load_multiple_files(
        frag_filters,
        columns,
        schema,
        *to_pandas_args,
    ):
        import pyarrow as pa

        from dask.base import apply, tokenize
        from dask.threaded import get

        token = tokenize(frag_filters, columns, schema)
        name = f"pq-file-{token}"
        dsk = {
            (name, i): (
                CudfReadParquetPyarrowFS._fragment_to_table,
                frag,
                filter,
                columns,
                schema,
            )
            for i, (frag, filter) in enumerate(frag_filters)
        }
        dsk[name] = (
            apply,
            pa.concat_tables,
            [list(dsk.keys())],
            {"promote_options": "permissive"},
        )
        return CudfReadParquetPyarrowFS._table_to_pandas(
            get(dsk, name),
            *to_pandas_args,
        )


class CudfReadParquetPyarrowFS(ReadParquetPyarrowFS):
    @functools.cached_property
    def _dataset_info(self):
        from dask_cudf.io.parquet import set_object_dtypes_from_pa_schema

        dataset_info = super()._dataset_info
        meta_pd = dataset_info["base_meta"]

        # Convert to cudf
        # (drop unsupported timezone information)
        for k, v in meta_pd.dtypes.items():
            if isinstance(v, pd.DatetimeTZDtype) and v.tz is not None:
                meta_pd[k] = meta_pd[k].dt.tz_localize(None)
        meta_cudf = cudf.from_pandas(meta_pd)

        # Re-set "object" dtypes to align with pa schema
        kwargs = dataset_info.get("kwargs", {})
        set_object_dtypes_from_pa_schema(
            meta_cudf,
            kwargs.get("schema", None),
        )

        dataset_info["base_meta"] = meta_cudf
        self.operands[type(self)._parameters.index("_dataset_info_cache")] = (
            dataset_info
        )
        return dataset_info

    @staticmethod
    def _table_to_pandas(
        table,
        index_name,
        arrow_to_pandas,
        dtype_backend,
        pyarrow_strings_enabled,
    ):
        df = cudf.DataFrame.from_arrow(table)
        if index_name is not None:
            df = df.set_index(index_name)
        return df

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        if isinstance(parent, CudfFusedParquetIO):
            return
        return parent.substitute(self, CudfFusedParquetIO(self))


class ToCudfBackend(Elemwise):
    # TODO: Inherit from ToBackend when rapids-dask-dependency
    # is pinned to dask>=2024.8.1
    _parameters = ["frame", "options"]
    _projection_passthrough = True
    _filter_passthrough = True
    _preserves_partitioning_information = True

    @staticmethod
    def operation(df, options):
        from dask_cudf.backends import to_cudf_dispatch

        return to_cudf_dispatch(df, **options)

    def _simplify_down(self):
        if isinstance(
            self.frame._meta, (cudf.DataFrame, cudf.Series, cudf.Index)
        ):
            # We already have cudf data
            return self.frame


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


# Temporary work-around for missing cudf + categorical support
# See: https://github.com/rapidsai/cudf/issues/11795
# TODO: Fix RepartitionQuantiles and remove this in cudf>24.06

_original_get_divisions = _shuffle_module._get_divisions


def _patched_get_divisions(frame, other, *args, **kwargs):
    # NOTE: The following two lines contains the "patch"
    # (we simply convert the partitioning column to pandas)
    if is_categorical_dtype(other._meta.dtype) and hasattr(
        other.frame._meta, "to_pandas"
    ):
        other = new_collection(other).to_backend("pandas")._expr

    # Call "original" function
    return _original_get_divisions(frame, other, *args, **kwargs)


_shuffle_module._get_divisions = _patched_get_divisions
