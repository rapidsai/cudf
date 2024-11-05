# Copyright (c) 2024, NVIDIA CORPORATION.

import functools
import itertools
import math
import warnings

import numpy as np
import pandas as pd
from dask_expr._expr import Elemwise
from dask_expr._util import _convert_to_list
from dask_expr.io.io import FusedIO, FusedParquetIO
from dask_expr.io.parquet import (
    FragmentWrapper,
    ReadParquetFSSpec,
    ReadParquetPyarrowFS,
)

from dask.dataframe.io.parquet.arrow import _filters_to_expression
from dask.dataframe.io.parquet.core import ParquetFunctionWrapper
from dask.utils import parse_bytes

import cudf

from dask_cudf import _deprecated_api

# Dask-expr imports CudfEngine from this module
from dask_cudf._legacy.io.parquet import CudfEngine  # noqa: F401


class NoOp(Elemwise):
    # Workaround - Always wrap read_parquet operations
    # in a NoOp to trigger tune_up optimizations.
    _parameters = ["frame"]
    _is_length_preserving = True
    _projection_passthrough = True
    _filter_passthrough = True
    _preserves_partitioning_information = True

    @staticmethod
    def operation(x):
        return x


class CudfReadParquetFSSpec(ReadParquetFSSpec):
    def approx_statistics(self):
        # Use a few files to approximate column-size statistics

        # Account for filters
        ds_filters = None
        if self.filters is not None:
            ds_filters = _filters_to_expression(self.filters)

        # Use average total_uncompressed_size of three files
        n_sample = 3
        column_sizes = {}
        for i, frag in enumerate(
            self._dataset_info["ds"].get_fragments(ds_filters)
        ):
            md = frag.metadata
            for rg in range(md.num_row_groups):
                row_group = md.row_group(rg)
                for col in range(row_group.num_columns):
                    column = row_group.column(col)
                    name = column.path_in_schema
                    if name not in column_sizes:
                        column_sizes[name] = np.zeros(n_sample, dtype="int64")
                    column_sizes[name][i] += column.total_uncompressed_size
            if (i + 1) >= n_sample:
                break

        # Reorganize stats to look like arrow-fs version
        return {
            "columns": [
                {
                    "path_in_schema": name,
                    "total_uncompressed_size": np.mean(sizes),
                }
                for name, sizes in column_sizes.items()
            ]
        }

    # ## OLD
    # @property
    # def _fusion_compression_factor(self):
    #     if self.operand("columns") is None:
    #         return 1
    #     nr_original_columns = max(len(self._dataset_info["schema"].names) - 1, 1)
    #     return max(
    #         len(_convert_to_list(self.operand("columns"))) / nr_original_columns, 0.001
    #     )

    @functools.cached_property
    def _fusion_compression_factor(self):
        blocksize = self.blocksize
        if blocksize is None or self.aggregate_files:
            # NOTE: We cannot fuse files *again* if
            # aggregate_files is True (this creates
            # too much OOM risk)
            return 1
        elif blocksize == "default":
            blocksize = "256MiB"

        approx_stats = self.approx_statistics()
        projected_size = 0
        col_op = self.operand("columns") or self.columns
        for col in approx_stats["columns"]:
            if col["path_in_schema"] in col_op or (
                (split_name := col["path_in_schema"].split("."))
                and split_name[0] in col_op
            ):
                projected_size += col["total_uncompressed_size"]

        if projected_size < 1:
            return 1

        aggregate_files = max(1, int(parse_bytes(blocksize) / projected_size))
        return max(1 / aggregate_files, 0.001)

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        if isinstance(parent, FusedIO):
            return
        return parent.substitute(self, CudfFusedIO(self))


class CudfReadParquetPyarrowFS(ReadParquetPyarrowFS):
    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "filesystem",
        "blocksize",
        "ignore_metadata_file",
        "calculate_divisions",
        "arrow_to_pandas",
        "pyarrow_strings_enabled",
        "kwargs",
        "_partitions",
        "_series",
        "_dataset_info_cache",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "filesystem": None,
        "blocksize": "256 MiB",
        "ignore_metadata_file": True,
        "calculate_divisions": False,
        "arrow_to_pandas": None,
        "pyarrow_strings_enabled": True,
        "kwargs": None,
        "_partitions": None,
        "_series": False,
        "_dataset_info_cache": None,
    }

    @functools.cached_property
    def _dataset_info(self):
        from dask_cudf._legacy.io.parquet import (
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
        if isinstance(table, cudf.DataFrame):
            df = table
        else:
            df = cudf.DataFrame.from_arrow(table)
            if index_name is not None:
                return df.set_index(index_name)
        return df

    @staticmethod
    def _fragments_to_cudf_dataframe(
        fragment_wrappers,
        filters,
        columns,
        schema,
    ):
        from dask.dataframe.io.utils import _is_local_fs

        from cudf.io.parquet import _apply_post_filters, _normalize_filters

        if not isinstance(fragment_wrappers, list):
            fragment_wrappers = [fragment_wrappers]

        filesystem = None
        paths, row_groups = [], []
        for fw in fragment_wrappers:
            frag = fw.fragment if isinstance(fw, FragmentWrapper) else fw
            paths.append(frag.path)
            row_groups.append(
                [rg.id for rg in frag.row_groups] if frag.row_groups else None
            )
            if filesystem is None:
                filesystem = frag.filesystem

        if _is_local_fs(filesystem):
            filesystem = None
        else:
            from fsspec.implementations.arrow import ArrowFSWrapper

            filesystem = ArrowFSWrapper(filesystem)
            protocol = filesystem.protocol
            paths = [f"{protocol}://{path}" for path in paths]

        filters = _normalize_filters(filters)
        projected_columns = None
        if columns and filters:
            projected_columns = [c for c in columns if c is not None]
            columns = sorted(
                set(v[0] for v in itertools.chain.from_iterable(filters))
                | set(projected_columns)
            )

        if row_groups == [None for path in paths]:
            row_groups = None

        df = cudf.read_parquet(
            paths,
            columns=columns,
            filters=filters,
            row_groups=row_groups,
            dataset_kwargs={"schema": schema},
        )

        # Apply filters (if any are defined)
        df = _apply_post_filters(df, filters)
        if projected_columns:
            # Elements of `projected_columns` may now be in the index.
            # We must filter these names from our projection
            projected_columns = [
                col for col in projected_columns if col in df._column_names
            ]
            df = df[projected_columns]

        # TODO: Deal with hive partitioning.
        # Note that ReadParquetPyarrowFS does NOT support this yet anyway.
        return df

    @functools.cached_property
    def _use_device_io(self):
        from dask.dataframe.io.utils import _is_local_fs

        # Use host for remote filesystem only
        return _is_local_fs(self.fs)
        # TODO: Use KvikIO-S3 support when available.
        # or (
        #     self.fs.type_name
        #     == "s3"  # TODO: and cudf.get_option("kvikio_remote_io")
        # )

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

        frag_to_table = self._fragment_to_table
        if self._use_device_io:
            frag_to_table = self._fragments_to_cudf_dataframe
        return (
            self._table_to_pandas,
            (
                frag_to_table,
                FragmentWrapper(self.fragments[index], filesystem=self.fs),
                self.filters,
                columns,
                schema,
            ),
            index_name,
        )

    @property
    def _fusion_compression_factor(self):
        blocksize = self.blocksize
        if blocksize is None:
            return 1
        elif blocksize == "default":
            blocksize = "256MiB"

        approx_stats = self.approx_statistics()
        projected_size = 0
        col_op = self.operand("columns") or self.columns
        for col in approx_stats["columns"]:
            if col["path_in_schema"] in col_op or (
                (split_name := col["path_in_schema"].split("."))
                and split_name[0] in col_op
            ):
                projected_size += col["total_uncompressed_size"]

        if projected_size < 1:
            return 1

        aggregate_files = max(1, int(parse_bytes(blocksize) / projected_size))
        return max(1 / aggregate_files, 0.001)

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        fused_cls = (
            CudfFusedParquetIO
            if self._use_device_io
            else CudfFusedParquetIOHost
        )
        if isinstance(parent, fused_cls):
            return
        return parent.substitute(self, fused_cls(self))


class CudfFusedIO(FusedIO):
    def _task(self, index: int):
        expr = self.operand("_expr")
        bucket = self._fusion_buckets[index]

        io_func = expr._filtered_task(0)[0]
        if not isinstance(
            io_func, ParquetFunctionWrapper
        ) or io_func.common_kwargs.get("partitions", None):
            # Just use "simple" fusion if we have an unexpected
            # callable, or we are dealing with hive partitioning.
            return (cudf.concat, [expr._filtered_task(i) for i in bucket])

        pieces = []
        for i in bucket:
            piece = expr._filtered_task(i)[1]
            if isinstance(piece, list):
                pieces.extend(piece)
            else:
                pieces.append(piece)
        return (io_func, pieces)


class CudfFusedParquetIO(FusedParquetIO):
    @functools.cached_property
    def _fusion_buckets(self):
        partitions = self.operand("_expr")._partitions
        npartitions = len(partitions)

        step = math.ceil(1 / self.operand("_expr")._fusion_compression_factor)

        # TODO: Heuristic to limit fusion should probably
        # account for the number of workers. For now, just
        # limiting fusion to 100 partitions at once.
        step = min(step, 100)

        buckets = [
            partitions[i : i + step] for i in range(0, npartitions, step)
        ]
        return buckets

    @classmethod
    def _load_multiple_files(
        cls,
        frag_filters,
        columns,
        schema,
        *to_pandas_args,
    ):
        frag_to_table = CudfReadParquetPyarrowFS._fragments_to_cudf_dataframe
        return CudfReadParquetPyarrowFS._table_to_pandas(
            frag_to_table(
                [frag[0] for frag in frag_filters],
                frag_filters[0][1],  # TODO: Check for consistent filters?
                columns,
                schema,
            ),
            *to_pandas_args,
        )


class CudfFusedParquetIOHost(CudfFusedParquetIO):
    @classmethod
    def _load_multiple_files(
        cls,
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


def read_parquet(
    path,
    *args,
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
    **kwargs,
):
    import dask_expr as dx
    from fsspec.utils import stringify_path
    from pyarrow import fs as pa_fs

    from dask.core import flatten
    from dask.dataframe.utils import pyarrow_strings_enabled

    from dask_cudf.backends import PYARROW_GE_15

    if args:
        raise ValueError(f"Unexpected positional arguments: {args}")

    if dtype_backend is not None:
        raise NotImplementedError(
            "dtype_backend is not supported by the 'cudf' backend."
        )
    if arrow_to_pandas is not None:
        raise NotImplementedError(
            "arrow_to_pandas is not supported by the 'cudf' backend."
        )
    if engine not in (None, "cudf", CudfEngine):
        raise NotImplementedError(
            "engine={engine} is not supported by the 'cudf' backend."
        )

    if not isinstance(path, str):
        path = stringify_path(path)

    kwargs["dtype_backend"] = None
    if arrow_to_pandas:
        kwargs["arrow_to_pandas"] = None

    if filters is not None:
        for filter in flatten(filters, container=list):
            _, op, val = filter
            if op == "in" and not isinstance(val, (set, list, tuple)):
                raise TypeError(
                    "Value of 'in' filter must be a list, set or tuple."
                )

    if (
        isinstance(filesystem, pa_fs.FileSystem)
        or isinstance(filesystem, str)
        and filesystem.lower() in ("arrow", "pyarrow")
    ):
        # EXPERIMENTAL filesystem="arrow" support.
        # This code path may use PyArrow for remote IO.

        # CudfReadParquetPyarrowFS requires import of distributed beforehand
        # (See: https://github.com/dask/dask/issues/11352)
        import distributed  # noqa: F401

        if not PYARROW_GE_15:
            raise ValueError(
                "pyarrow>=15.0.0 is required to use the pyarrow filesystem."
            )
        if metadata_task_size is not None:
            warnings.warn(
                "metadata_task_size is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if aggregate_files is not None:
            warnings.warn(
                "aggregate_files is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if split_row_groups != "infer":
            warnings.warn(
                "split_row_groups is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if parquet_file_extension != (".parq", ".parquet", ".pq"):
            raise NotImplementedError(
                "parquet_file_extension is not supported when using the pyarrow filesystem."
            )

        return dx.new_collection(
            NoOp(
                CudfReadParquetPyarrowFS(
                    path,
                    columns=_convert_to_list(columns),
                    filters=filters,
                    categories=categories,
                    index=index,
                    calculate_divisions=calculate_divisions,
                    storage_options=storage_options,
                    filesystem=filesystem,
                    blocksize=blocksize,
                    ignore_metadata_file=ignore_metadata_file,
                    arrow_to_pandas=None,
                    pyarrow_strings_enabled=pyarrow_strings_enabled(),
                    kwargs=kwargs,
                    _series=isinstance(columns, str),
                ),
            )
        )

    return dx.new_collection(
        NoOp(
            CudfReadParquetFSSpec(
                path,
                columns=_convert_to_list(columns),
                filters=filters,
                categories=categories,
                index=index,
                blocksize=blocksize,
                storage_options=storage_options,
                calculate_divisions=calculate_divisions,
                ignore_metadata_file=ignore_metadata_file,
                metadata_task_size=metadata_task_size,
                split_row_groups=split_row_groups,
                aggregate_files=aggregate_files,
                parquet_file_extension=parquet_file_extension,
                filesystem=filesystem,
                engine=CudfEngine,
                kwargs=kwargs,
                _series=isinstance(columns, str),
            ),
        )
    )


to_parquet = _deprecated_api(
    "dask_cudf.io.parquet.to_parquet",
    new_api="dask_cudf._legacy.io.parquet.to_parquet",
    rec="Please use the DataFrame.to_parquet method instead.",
)
create_metadata_file = _deprecated_api(
    "dask_cudf.io.parquet.create_metadata_file",
    new_api="dask_cudf._legacy.io.parquet.create_metadata_file",
    rec="Please raise an issue if this feature is needed.",
)
