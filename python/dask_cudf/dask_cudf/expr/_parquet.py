# Copyright (c) 2024, NVIDIA CORPORATION.

from functools import cached_property

import dask_expr as dx
import pandas as pd
import pyarrow as pa
from dask_expr.io.io import FusedParquetIO
from dask_expr.io.parquet import ReadParquetPyarrowFS
from fsspec.utils import stringify_path
from pyarrow import fs as pa_fs

from dask import config
from dask.base import apply, tokenize
from dask.core import flatten
from dask.threaded import get

import cudf

from dask_cudf.io.parquet import CudfEngine, set_object_dtypes_from_pa_schema


class CudfFusedParquetIO(FusedParquetIO):
    @staticmethod
    def _load_multiple_files(
        frag_filters,
        columns,
        schema,
        *to_pandas_args,
    ):
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
    @cached_property
    def _dataset_info(self):
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


def _read_parquet(
    path=None,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    dtype_backend=None,
    calculate_divisions=False,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem="fsspec",
    engine=None,
    arrow_to_pandas=None,
    open_file_options=None,
    **kwargs,
):
    if not isinstance(path, str):
        path = stringify_path(path)

    if dtype_backend is not None:
        raise NotImplementedError()
    if arrow_to_pandas is not None:
        raise NotImplementedError()
    if open_file_options is not None:
        raise NotImplementedError()

    if filters is not None:
        for filter in flatten(filters, container=list):
            col, op, val = filter
            if op == "in" and not isinstance(val, (set, list, tuple)):
                raise TypeError(
                    "Value of 'in' filter must be a list, set or tuple."
                )

    if (
        isinstance(filesystem, pa_fs.FileSystem)
        or isinstance(filesystem, str)
        and filesystem.lower() in ("arrow", "pyarrow")
    ):
        if metadata_task_size is not None:
            raise NotImplementedError(
                "metadata_task_size is not supported when using the pyarrow filesystem."
            )
        if split_row_groups != "infer":
            raise NotImplementedError(
                "split_row_groups is not supported when using the pyarrow filesystem."
            )
        if blocksize is not None and blocksize != "default":
            raise NotImplementedError(
                "blocksize is not supported when using the pyarrow filesystem."
            )
        if aggregate_files is not None:
            raise NotImplementedError(
                "aggregate_files is not supported when using the pyarrow filesystem."
            )
        if parquet_file_extension != (".parq", ".parquet", ".pq"):
            raise NotImplementedError(
                "parquet_file_extension is not supported when using the pyarrow filesystem."
            )
        if engine is not None:
            raise NotImplementedError(
                "engine is not supported when using the pyarrow filesystem."
            )

        return dx.new_collection(
            CudfReadParquetPyarrowFS(
                path,
                columns=dx._util._convert_to_list(columns),
                filters=filters,
                categories=categories,
                index=index,
                calculate_divisions=calculate_divisions,
                storage_options=storage_options,
                filesystem=filesystem,
                ignore_metadata_file=ignore_metadata_file,
                arrow_to_pandas=arrow_to_pandas,
                pyarrow_strings_enabled=False,
                kwargs=kwargs,
                _series=isinstance(columns, str),
            )
        )
    else:
        with config.set({"dataframe.backend": "pandas"}):
            return dx.read_parquet(
                path=path,
                columns=columns,
                filters=filters,
                categories=categories,
                index=index,
                storage_options=storage_options,
                calculate_divisions=calculate_divisions,
                ignore_metadata_file=ignore_metadata_file,
                metadata_task_size=metadata_task_size,
                split_row_groups=split_row_groups,
                blocksize=blocksize,
                aggregate_files=aggregate_files,
                parquet_file_extension=parquet_file_extension,
                filesystem=filesystem,
                engine=CudfEngine,
                **kwargs,
            )
