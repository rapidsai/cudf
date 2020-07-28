# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import warnings
from functools import partial
from io import BufferedWriter, IOBase

from dask import dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

import cudf
from cudf.core.column import as_column, build_categorical_column
from cudf.io import write_to_dataset


class CudfEngine(ArrowEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        meta, stats, parts, index = ArrowEngine.read_metadata(*args, **kwargs)

        # If `strings_to_categorical==True`, convert objects to int32
        strings_to_cats = kwargs.get("strings_to_categorical", False)

        new_meta = cudf.DataFrame(index=meta.index)
        for col in meta.columns:
            if meta[col].dtype == "O":
                new_meta[col] = as_column(
                    meta[col], dtype="int32" if strings_to_cats else "object"
                )
            else:
                new_meta[col] = as_column(meta[col])

        return (new_meta, stats, parts, index)

    @staticmethod
    def read_partition(
        fs, piece, columns, index, categories=(), partitions=(), **kwargs
    ):
        if columns is not None:
            columns = [c for c in columns]
        if isinstance(index, list):
            columns += index

        if isinstance(piece, str):
            path = piece
            row_group = None
            partition_keys = []
        else:
            (path, row_group, partition_keys) = piece

        strings_to_cats = kwargs.get("strings_to_categorical", False)
        if cudf.utils.ioutils._is_local_filesystem(fs):
            df = cudf.read_parquet(
                path,
                engine="cudf",
                columns=columns,
                row_groups=row_group,
                strings_to_categorical=strings_to_cats,
                **kwargs.get("read", {}),
            )
        else:
            with fs.open(path, mode="rb") as f:
                df = cudf.read_parquet(
                    f,
                    engine="cudf",
                    columns=columns,
                    row_groups=row_group,
                    strings_to_categorical=strings_to_cats,
                    **kwargs.get("read", {}),
                )

        if index and (index[0] in df.columns):
            df = df.set_index(index[0])
        if partition_keys:
            if partitions is None:
                raise ValueError("Must pass partition sets")
            for i, (name, index2) in enumerate(partition_keys):
                categories = [
                    val.as_py() for val in partitions.levels[i].dictionary
                ]

                col = as_column(index2).as_frame().repeat(len(df))._data[None]
                df[name] = build_categorical_column(
                    categories=categories,
                    codes=as_column(col.base_data, dtype=col.dtype),
                    size=col.size,
                    offset=col.offset,
                    ordered=False,
                )

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
        compression=None,
        index_cols=None,
        **kwargs,
    ):
        preserve_index = False
        if partition_on:
            md = write_to_dataset(
                df,
                path,
                partition_cols=partition_on,
                fs=fs,
                preserve_index=preserve_index,
                return_metadata=return_metadata,
                **kwargs,
            )
        else:
            with fs.open(fs.sep.join([path, filename]), mode="wb") as out_file:
                if not isinstance(out_file, IOBase):
                    out_file = BufferedWriter(out_file)
                md = df.to_parquet(
                    out_file,
                    compression=compression,
                    metadata_file_path=filename if return_metadata else None,
                    **kwargs,
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


def read_parquet(
    path,
    columns=None,
    split_row_groups=None,
    row_groups_per_part=None,
    **kwargs,
):
    """ Read parquet files into a Dask DataFrame

    Calls ``dask.dataframe.read_parquet`` to cordinate the execution of
    ``cudf.read_parquet``, and ultimately read multiple partitions into a
    single Dask dataframe. The Dask version must supply an ``ArrowEngine``
    class to support full functionality.
    See ``cudf.read_parquet`` and Dask documentation for further details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_parquet("/path/to/dataset/")  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """
    if isinstance(columns, str):
        columns = [columns]

    if row_groups_per_part:
        warnings.warn(
            "row_groups_per_part is deprecated. "
            "Pass an integer value to split_row_groups instead."
        )
        if split_row_groups is None:
            split_row_groups = row_groups_per_part

    return dd.read_parquet(
        path,
        columns=columns,
        split_row_groups=split_row_groups,
        engine=CudfEngine,
        **kwargs,
    )


to_parquet = partial(dd.to_parquet, engine=CudfEngine)
