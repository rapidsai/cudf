# Copyright (c) 2020, NVIDIA CORPORATION.

import warnings

import pyarrow.parquet as pq
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path

from dask.base import tokenize
from dask.dataframe.io.parquet.core import set_index_columns
from dask.dataframe.io.parquet.utils import (
    _normalize_index_columns,
    _parse_pandas_metadata,
)
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes

import cudf
from cudf.core.column import as_column

from dask_cudf import DataFrame

try:
    import ujson as json
except ImportError:
    import json


def _get_dataset_and_parts(data_path, fs, row_groups_per_part):
    parts = []
    dataset = pq.ParquetDataset(data_path, filesystem=fs)
    if dataset.metadata:
        fpath_last = None
        rgi = 0
        rg_list = []
        for rg in range(dataset.metadata.num_row_groups):

            fpath = dataset.metadata.row_group(rg).column(0).file_path

            if fpath_last and fpath_last != fpath:
                rgi = 0
                full_path = fs.sep.join([data_path, fpath_last])
                parts.append(tuple([full_path, rg_list]))
                rg_list = []
            elif len(rg_list) >= row_groups_per_part:
                full_path = fs.sep.join([data_path, fpath_last])
                parts.append(tuple([full_path, rg_list]))
                rg_list = []

            if fpath is None:
                raise ValueError("_metadata file is missing file_path string.")

            fpath_last = fpath
            rg_list.append(rgi)
            rgi += 1
        if rg_list:
            full_path = fs.sep.join([data_path, fpath_last])
            parts.append(tuple([full_path, rg_list]))
    else:
        warnings.warn(
            "Must have metadata file to split by row group."
            "Using full file for each partition."
        )
        for piece in dataset.pieces:
            parts.append(tuple([piece.path, None]))

    return dataset, parts


def _read_metadata(fs, path, row_groups_per_part, index=None):
    dataset, parts = _get_dataset_and_parts(path, fs, row_groups_per_part)
    if not dataset.metadata:
        raise ValueError("_metadata file is missing.")

    schema = dataset.metadata.schema.to_arrow_schema()
    columns = None
    has_pandas_metadata = (
        schema.metadata is not None and b"pandas" in schema.metadata
    )
    categories = None
    if has_pandas_metadata:
        pandas_metadata = json.loads(schema.metadata[b"pandas"].decode("utf8"))
        (
            index_names,
            column_names,
            storage_name_mapping,
            column_index_names,
        ) = _parse_pandas_metadata(pandas_metadata)
        categories = []
        for col in pandas_metadata["columns"]:
            if (col["pandas_type"] == "categorical") and (
                col["name"] not in categories
            ):
                categories.append(col["name"])
    else:
        index_names = []
        column_names = schema.names
        storage_name_mapping = {k: k for k in column_names}
        column_index_names = [None]

    if index is None and index_names:
        index = index_names

    column_names, index_names = _normalize_index_columns(
        columns, column_names, index, index_names
    )
    all_columns = index_names + column_names

    dtypes = _get_pyarrow_dtypes(schema, categories)
    dtypes = {storage_name_mapping.get(k, k): v for k, v in dtypes.items()}

    index_cols = index or ()
    meta = _meta_from_dtypes(
        all_columns, dtypes, index_cols, column_index_names
    )

    return meta, parts


def _read_partition(part, index, columns, strings_to_cats):
    # Read dataset part
    path, row_groups = part
    if columns is not None:
        columns = [c for c in columns]
    if isinstance(index, list):
        columns += index

    df = cudf.io.read_parquet(
        path,
        row_groups=row_groups,
        columns=columns,
        strings_to_cats=strings_to_cats,
    )

    if index and (index[0] in df.columns):
        df = df.set_index(index[0])
    return df


def parquet_reader(
    path,
    columns=None,
    row_groups_per_part=None,
    index=None,
    storage_options=None,
    **kwargs,
):

    name = "opt-read-parquet-" + tokenize(
        path, columns, index, storage_options, row_groups_per_part
    )

    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    if len(paths) > 1 or not fs.isdir(paths[0]):
        raise ValueError(
            "Must pass in a directory path to use `row_groups_per_part`."
        )

    auto_index_allowed = False
    if index is None:
        # User is allowing auto-detected index
        auto_index_allowed = True
    if index and isinstance(index, str):
        index = [index]

    dd_meta, parts = _read_metadata(fs, path, row_groups_per_part, index=index)
    strings_to_cats = kwargs.get("strings_to_categorical", False)
    meta = cudf.DataFrame(index=dd_meta.index)
    for col in dd_meta.columns:
        if dd_meta[col].dtype == "O":
            meta[col] = as_column(
                dd_meta[col], dtype="int32" if strings_to_cats else "object"
            )
        else:
            meta[col] = as_column(dd_meta[col])

    if meta.index.name is not None:
        index = meta.index.name

    # Account for index and columns arguments.
    # Modify `meta` dataframe accordingly
    index_in_columns = False
    meta, index, columns = set_index_columns(
        meta, index, columns, index_in_columns, auto_index_allowed
    )

    dsk = {}
    for p, part in enumerate(parts):
        read_key = (name, p)
        dsk[read_key] = (
            _read_partition,
            part,
            index,
            columns,
            strings_to_cats,
        )

    # Set the index that was previously treated as a column
    if index_in_columns:
        meta = meta.set_index(index)

    divisions = [None] * (len(parts) + 1)
    return DataFrame(dsk, name, meta, divisions)
