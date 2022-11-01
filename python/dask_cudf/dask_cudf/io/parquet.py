# Copyright (c) 2019-2022, NVIDIA CORPORATION.
import warnings
from contextlib import ExitStack
from functools import partial
from io import BufferedWriter, BytesIO, IOBase

import numpy as np
from pyarrow import dataset as pa_ds, parquet as pq

from dask import dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine

try:
    from dask.dataframe.io.parquet import (
        create_metadata_file as create_metadata_file_dd,
    )
except ImportError:
    create_metadata_file_dd = None

import cudf
from cudf.core.column import as_column, build_categorical_column
from cudf.io import write_to_dataset
from cudf.io.parquet import _default_open_file_options
from cudf.utils.dtypes import cudf_dtype_from_pa_type
from cudf.utils.ioutils import (
    _ROW_GROUP_SIZE_BYTES_DEFAULT,
    _is_local_filesystem,
    _open_remote_files,
)


class CudfEngine(ArrowDatasetEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        meta, stats, parts, index = ArrowDatasetEngine.read_metadata(
            *args, **kwargs
        )
        new_meta = cudf.from_pandas(meta)
        if parts:
            # Re-set "object" dtypes align with pa schema
            set_object_dtypes_from_pa_schema(
                new_meta,
                parts[0].get("common_kwargs", {}).get("schema", None),
            )

        # If `strings_to_categorical==True`, convert objects to int32
        strings_to_cats = kwargs.get("strings_to_categorical", False)
        for col in new_meta._data.names:
            if (
                isinstance(new_meta._data[col], cudf.core.column.StringColumn)
                and strings_to_cats
            ):
                new_meta._data[col] = new_meta._data[col].astype("int32")
        return (new_meta, stats, parts, index)

    @classmethod
    def multi_support(cls):
        # Assert that this class is CudfEngine
        # and that multi-part reading is supported
        return cls == CudfEngine

    @classmethod
    def _read_paths(
        cls,
        paths,
        fs,
        columns=None,
        row_groups=None,
        strings_to_categorical=None,
        partitions=None,
        partitioning=None,
        partition_keys=None,
        open_file_options=None,
        **kwargs,
    ):

        # Simplify row_groups if all None
        if row_groups == [None for path in paths]:
            row_groups = None

        with ExitStack() as stack:

            # Non-local filesystem handling
            paths_or_fobs = paths
            if not _is_local_filesystem(fs):
                paths_or_fobs = _open_remote_files(
                    paths_or_fobs,
                    fs,
                    context_stack=stack,
                    **_default_open_file_options(
                        open_file_options, columns, row_groups
                    ),
                )

            # Use cudf to read in data
            df = cudf.read_parquet(
                paths_or_fobs,
                engine="cudf",
                columns=columns,
                row_groups=row_groups if row_groups else None,
                strings_to_categorical=strings_to_categorical,
                **kwargs,
            )

        if partitions and partition_keys is None:

            # Use `HivePartitioning` by default
            partitioning = partitioning or {"obj": pa_ds.HivePartitioning}
            ds = pa_ds.dataset(
                paths,
                filesystem=fs,
                format="parquet",
                partitioning=partitioning["obj"].discover(
                    *partitioning.get("args", []),
                    **partitioning.get("kwargs", {}),
                ),
            )
            frag = next(ds.get_fragments())
            if frag:
                # Extract hive-partition keys, and make sure they
                # are ordered the same as they are in `partitions`
                raw_keys = pa_ds._get_partition_keys(frag.partition_expression)
                partition_keys = [
                    (hive_part.name, raw_keys[hive_part.name])
                    for hive_part in partitions
                ]

        if partition_keys:
            if partitions is None:
                raise ValueError("Must pass partition sets")

            for i, (name, index2) in enumerate(partition_keys):

                # Build the column from `codes` directly
                # (since the category is often a larger dtype)
                codes = as_column(
                    partitions[i].keys.index(index2),
                    length=len(df),
                )
                df[name] = build_categorical_column(
                    categories=partitions[i].keys,
                    codes=codes,
                    size=codes.size,
                    offset=codes.offset,
                    ordered=False,
                )

        return df

    @classmethod
    def read_partition(
        cls,
        fs,
        pieces,
        columns,
        index,
        categories=(),
        partitions=(),
        partitioning=None,
        schema=None,
        open_file_options=None,
        **kwargs,
    ):

        if columns is not None:
            columns = [c for c in columns]
        if isinstance(index, list):
            columns += index

        # Check if we are actually selecting any columns
        read_columns = columns
        if schema and columns:
            ignored = set(schema.names) - set(columns)
            if not ignored:
                read_columns = None

        if not isinstance(pieces, list):
            pieces = [pieces]

        # Extract supported kwargs from `kwargs`
        strings_to_cats = kwargs.get("strings_to_categorical", False)
        read_kwargs = kwargs.get("read", {})
        read_kwargs.update(open_file_options or {})
        check_file_size = read_kwargs.pop("check_file_size", None)

        # Wrap reading logic in a `try` block so that we can
        # inform the user that the `read_parquet` partition
        # size is too large for the available memory
        try:

            # Assume multi-piece read
            paths = []
            rgs = []
            last_partition_keys = None
            dfs = []

            for i, piece in enumerate(pieces):

                (path, row_group, partition_keys) = piece
                row_group = None if row_group == [None] else row_group

                # File-size check to help "protect" users from change
                # to up-stream `split_row_groups` default. We only
                # check the file size if this partition corresponds
                # to a full file, and `check_file_size` is defined
                if check_file_size and len(pieces) == 1 and row_group is None:
                    file_size = fs.size(path)
                    if file_size > check_file_size:
                        warnings.warn(
                            f"A large parquet file ({file_size}B) is being "
                            f"used to create a DataFrame partition in "
                            f"read_parquet. This may cause out of memory "
                            f"exceptions in operations downstream. See the "
                            f"notes on split_row_groups in the read_parquet "
                            f"documentation. Setting split_row_groups "
                            f"explicitly will silence this warning."
                        )

                if i > 0 and partition_keys != last_partition_keys:
                    dfs.append(
                        cls._read_paths(
                            paths,
                            fs,
                            columns=read_columns,
                            row_groups=rgs if rgs else None,
                            strings_to_categorical=strings_to_cats,
                            partitions=partitions,
                            partitioning=partitioning,
                            partition_keys=last_partition_keys,
                            **read_kwargs,
                        )
                    )
                    paths = rgs = []
                    last_partition_keys = None
                paths.append(path)
                rgs.append(
                    [row_group]
                    if not isinstance(row_group, list)
                    and row_group is not None
                    else row_group
                )
                last_partition_keys = partition_keys

            dfs.append(
                cls._read_paths(
                    paths,
                    fs,
                    columns=read_columns,
                    row_groups=rgs if rgs else None,
                    strings_to_categorical=strings_to_cats,
                    partitions=partitions,
                    partitioning=partitioning,
                    partition_keys=last_partition_keys,
                    **read_kwargs,
                )
            )
            df = cudf.concat(dfs) if len(dfs) > 1 else dfs[0]

            # Re-set "object" dtypes align with pa schema
            set_object_dtypes_from_pa_schema(df, schema)

            if index and (index[0] in df.columns):
                df = df.set_index(index[0])
            elif index is False and df.index.names != (None,):
                # If index=False, we shouldn't have a named index
                df.reset_index(inplace=True)

        except MemoryError as err:
            raise MemoryError(
                "Parquet data was larger than the available GPU memory!\n\n"
                "See the notes on split_row_groups in the read_parquet "
                "documentation.\n\n"
                "Original Error: " + str(err)
            )
            raise err

        return df

    @staticmethod
    def write_partition(
        df,
        path,
        fs,
        filename,
        partition_on,
        return_metadata,
        fmd=None,
        compression="snappy",
        index_cols=None,
        **kwargs,
    ):
        preserve_index = False
        if set(index_cols).issubset(set(df.columns)):
            df.index = df[index_cols].copy(deep=False)
            df.drop(columns=index_cols, inplace=True)
            preserve_index = True
        if partition_on:
            md = write_to_dataset(
                df=df,
                root_path=path,
                compression=compression,
                filename=filename,
                partition_cols=partition_on,
                fs=fs,
                preserve_index=preserve_index,
                return_metadata=return_metadata,
                statistics=kwargs.get("statistics", "ROWGROUP"),
                int96_timestamps=kwargs.get("int96_timestamps", False),
                row_group_size_bytes=kwargs.get(
                    "row_group_size_bytes", _ROW_GROUP_SIZE_BYTES_DEFAULT
                ),
                row_group_size_rows=kwargs.get("row_group_size_rows", None),
                max_page_size_bytes=kwargs.get("max_page_size_bytes", None),
                max_page_size_rows=kwargs.get("max_page_size_rows", None),
                storage_options=kwargs.get("storage_options", None),
            )
        else:
            with fs.open(fs.sep.join([path, filename]), mode="wb") as out_file:
                if not isinstance(out_file, IOBase):
                    out_file = BufferedWriter(out_file)
                md = df.to_parquet(
                    path=out_file,
                    engine=kwargs.get("engine", "cudf"),
                    index=kwargs.get("index", None),
                    partition_cols=kwargs.get("partition_cols", None),
                    partition_file_name=kwargs.get(
                        "partition_file_name", None
                    ),
                    partition_offsets=kwargs.get("partition_offsets", None),
                    statistics=kwargs.get("statistics", "ROWGROUP"),
                    int96_timestamps=kwargs.get("int96_timestamps", False),
                    row_group_size_bytes=kwargs.get(
                        "row_group_size_bytes", _ROW_GROUP_SIZE_BYTES_DEFAULT
                    ),
                    row_group_size_rows=kwargs.get(
                        "row_group_size_rows", None
                    ),
                    storage_options=kwargs.get("storage_options", None),
                    metadata_file_path=filename if return_metadata else None,
                )
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"meta": md}]
        else:
            return []

    @staticmethod
    def write_metadata(parts, fmd, fs, path, append=False, **kwargs):
        if parts:
            # Aggregate metadata and write to _metadata file
            metadata_path = fs.sep.join([path, "_metadata"])
            _meta = []
            if append and fmd is not None:
                _meta = [fmd]
            _meta.extend([parts[i][0]["meta"] for i in range(len(parts))])
            _meta = (
                cudf.io.merge_parquet_filemetadata(_meta)
                if len(_meta) > 1
                else _meta[0]
            )
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(_meta))

    @classmethod
    def collect_file_metadata(cls, path, fs, file_path):
        with fs.open(path, "rb") as f:
            meta = pq.ParquetFile(f).metadata
        if file_path:
            meta.set_file_path(file_path)
        with BytesIO() as myio:
            meta.write_metadata_file(myio)
            myio.seek(0)
            meta = np.frombuffer(myio.read(), dtype="uint8")
        return meta

    @classmethod
    def aggregate_metadata(cls, meta_list, fs, out_path):
        meta = (
            cudf.io.merge_parquet_filemetadata(meta_list)
            if len(meta_list) > 1
            else meta_list[0]
        )
        if out_path:
            metadata_path = fs.sep.join([out_path, "_metadata"])
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(meta))
            return None
        else:
            return meta


def set_object_dtypes_from_pa_schema(df, schema):
    # Simple utility to modify cudf DataFrame
    # "object" dtypes to agree with a specific
    # pyarrow schema.
    if schema:
        for col_name, col in df._data.items():
            if col_name is None:
                # Pyarrow cannot handle `None` as a field name.
                # However, this should be a simple range index that
                # we can ignore anyway
                continue
            typ = cudf_dtype_from_pa_type(schema.field(col_name).type)
            if (
                col_name in schema.names
                and not isinstance(typ, (cudf.ListDtype, cudf.StructDtype))
                and isinstance(col, cudf.core.column.StringColumn)
            ):
                df._data[col_name] = col.astype(typ)


def read_parquet(path, columns=None, **kwargs):
    """Read parquet files into a Dask DataFrame

    Calls ``dask.dataframe.read_parquet`` with ``engine=CudfEngine``
    to coordinate the execution of ``cudf.read_parquet``, and to
    ultimately create a ``dask_cudf.DataFrame`` collection.

    See the ``dask.dataframe.read_parquet`` documentation for
    all available options.

    Examples
    --------
    >>> from dask_cudf import read_parquet
    >>> df = read_parquet("/path/to/dataset/")  # doctest: +SKIP

    When dealing with one or more large parquet files having an
    in-memory footprint >15% device memory, the ``split_row_groups``
    argument should be used to map Parquet **row-groups** to DataFrame
    partitions (instead of **files** to partitions). For example, the
    following code will map each row-group to a distinct partition:

    >>> df = read_parquet(..., split_row_groups=True)  # doctest: +SKIP

    To map **multiple** row-groups to each partition, an integer can be
    passed to ``split_row_groups`` to specify the **maximum** number of
    row-groups allowed in each output partition:

    >>> df = read_parquet(..., split_row_groups=10)  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """
    if isinstance(columns, str):
        columns = [columns]

    # Set "check_file_size" option to determine whether we
    # should check the parquet-file size. This check is meant
    # to "protect" users from `split_row_groups` default changes
    check_file_size = kwargs.pop("check_file_size", 500_000_000)
    if (
        check_file_size
        and ("split_row_groups" not in kwargs)
        and ("chunksize" not in kwargs)
    ):
        # User is not specifying `split_row_groups` or `chunksize`,
        # so we should warn them if/when a file is ~>0.5GB on disk.
        # They can set `split_row_groups` explicitly to silence/skip
        # this check
        if "read" not in kwargs:
            kwargs["read"] = {}
        kwargs["read"]["check_file_size"] = check_file_size

    return dd.read_parquet(path, columns=columns, engine=CudfEngine, **kwargs)


to_parquet = partial(dd.to_parquet, engine=CudfEngine)

if create_metadata_file_dd is None:
    create_metadata_file = create_metadata_file_dd
else:
    create_metadata_file = partial(create_metadata_file_dd, engine=CudfEngine)
