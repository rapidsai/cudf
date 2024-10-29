# Copyright (c) 2018-2024, NVIDIA CORPORATION.
import functools

import pandas as pd
from dask_expr.io.io import FusedParquetIO
from dask_expr.io.parquet import FragmentWrapper, ReadParquetPyarrowFS

import cudf

# Dask-expr imports CudfEngine from this module
from dask_cudf.legacy.io.parquet import CudfEngine  # noqa: F401


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
        from dask_cudf.legacy.io.parquet import (
            set_object_dtypes_from_pa_schema,
        )

        dataset_info = super()._dataset_info
        meta_pd = dataset_info["base_meta"]
        if isinstance(meta_pd, cudf.DataFrame):
            return dataset_info

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
    def _table_to_pandas(table, index_name):
        df = cudf.DataFrame.from_arrow(table)
        if index_name is not None:
            df = df.set_index(index_name)
        return df

    def _filtered_task(self, index: int):
        columns = self.columns.copy()
        index_name = self.index.name
        if self.index is not None:
            index_name = self.index.name
        schema = self._dataset_info["schema"].remove_metadata()
        if index_name:
            if columns is None:
                columns = list(schema.names)
            columns.append(index_name)
        return (
            self._table_to_pandas,
            (
                self._fragment_to_table,
                FragmentWrapper(self.fragments[index], filesystem=self.fs),
                self.filters,
                columns,
                schema,
            ),
            index_name,
        )

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        if isinstance(parent, CudfFusedParquetIO):
            return
        return parent.substitute(self, CudfFusedParquetIO(self))
