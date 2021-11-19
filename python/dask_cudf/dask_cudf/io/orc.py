# Copyright (c) 2020, NVIDIA CORPORATION.

from io import BufferedWriter, IOBase

from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from pyarrow import orc as orc

from dask import dataframe as dd, delayed
from dask.base import tokenize
from dask.dataframe.io.utils import _get_pyarrow_dtypes

import cudf


def _read_orc_stripe(fs, path, stripe, columns, kwargs=None):
    """Pull out specific columns from specific stripe"""
    if kwargs is None:
        kwargs = {}
    with fs.open(path, "rb") as f:
        df_stripe = cudf.read_orc(
            f, stripes=[stripe], columns=columns, **kwargs
        )
    return df_stripe


def _prepare_filters_with_cache(filters):
    # Coerce filters into list of lists of tuples
    if isinstance(filters[0][0], str):
        filters = [filters]

    return filters


def _prepare_filters(filters):
    return _prepare_filters_with_cache(
        tuple([tuple(conjunction) for conjunction in filters])
    )


def _filters_to_query(filters):
    query_string = ""
    local_dict = {}

    is_first_conjunction = True
    for conjunction in filters:
        # Generate or
        if is_first_conjunction:
            is_first_conjunction = False
        else:
            query_string += " or "

        # Generate string for conjunction
        query_string += "("
        for i, (col, op, val) in enumerate(conjunction):
            if i > 0:
                query_string += " and "
            query_string += "("
            # TODO: Add backticks around column name when cuDF query
            # function supports them
            query_string += col + " " + op + " @var" + str(i)
            query_string += ")"
            local_dict["var" + str(i)] = val
        query_string += ")"

    return query_string, local_dict


def read_orc(
    path,
    columns=None,
    filters=None,
    storage_options=None,
    filtering_columns_first=False,
    **kwargs,
):
    """Read cudf dataframe from ORC file(s).

    Note that this function is mostly borrowed from upstream Dask.

    Parameters
    ----------
    path: str or list(str)
        Location of file(s), which can be a full URL with protocol specifier,
        and may include glob character if a single string.
    columns: None or list(str)
        Columns to load. If None, loads all.
    filters : None or list of tuple or list of lists of tuples
        If not None, specifies a filter predicate used to filter out row groups
        using statistics stored for each row group as Parquet metadata. Row
        groups that do not match the given filter predicate are not read. The
        predicate is expressed in disjunctive normal form (DNF) like
        `[[('x', '=', 0), ...], ...]`. DNF allows arbitrary boolean logical
        combinations of single column predicates. The innermost tuples each
        describe a single column predicate. The list of inner predicates is
        interpreted as a conjunction (AND), forming a more selective and
        multiple column predicate. Finally, the outermost list combines
        these filters as a disjunction (OR). Predicates may also be passed
        as a list of tuples. This form is interpreted as a single conjunction.
        To express OR in predicates, one must use the (preferred) notation of
        list of lists of tuples.
    filtering_columns_first : bool, default False
        If True, reads in the columns referenced in `filters` first and uses
        that to determine what are the relevant stripes to read from other
        columns.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.

    Returns
    -------
    cudf.DataFrame
    """

    storage_options = storage_options or {}
    fs, fs_token, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    schema = None
    nstripes_per_file = []
    for path in paths:
        with fs.open(path, "rb") as f:
            o = orc.ORCFile(f)
            if schema is None:
                schema = o.schema
            elif schema != o.schema:
                raise ValueError(
                    "Incompatible schemas while parsing ORC files"
                )
            nstripes_per_file.append(o.nstripes)
    schema = _get_pyarrow_dtypes(schema, categories=None)
    if columns is not None:
        ex = set(columns) - set(schema)
        if ex:
            raise ValueError(
                "Requested columns (%s) not in schema (%s)" % (ex, set(schema))
            )
    else:
        columns = list(schema)

    # Read in filtering columns first
    if filters is None:
        filtering_columns_first = False
    if filtering_columns_first:
        # Determine which columns are filtering vs. remaining
        filters = _prepare_filters(filters)
        query_string, local_dict = _filters_to_query(filters)
        columns_in_predicate = list(
            {col for conjunction in filters for (col, _, _) in conjunction}
        )
        all_columns = columns
        columns = [c for c in all_columns if c not in columns_in_predicate]

        # Read in only the columns relevant to the filtering
        filtered_df = read_orc(
            paths,
            columns=columns_in_predicate,
            filters=filters,
            storage_options=None,
            **kwargs,
        )

        # Since the call to `read_orc` results in a partition for each relevant
        # stripe, we can simply check which partitions are empty. Then we can
        # read in only those relevant stripes.
        def _empty(df):
            if len(df.query(query_string, local_dict=local_dict)) == 0:
                return df.iloc[0:0, :].copy()
            else:
                return df

        filtered_df = filtered_df.map_partitions(_empty)
        filtered_partition_lens = filtered_df.map_partitions(len).compute()
        is_filtered_partition_empty = [
            filtered_partition_len == 0
            for filtered_partition_len in filtered_partition_lens
        ]

        # Cull empty partitions
        filtered_df_partitions = [
            filtered_df_partition
            for i, filtered_df_partition in enumerate(filtered_df.to_delayed())
            if not is_filtered_partition_empty[i]
        ]

    with fs.open(paths[0], "rb") as f:
        meta = cudf.read_orc(
            f,
            stripes=[0] if nstripes_per_file[0] else None,
            columns=columns,
            **kwargs,
        )

    name = "read-orc-" + tokenize(fs_token, path, columns, **kwargs)
    dsk = {}
    N = 0
    partition_idx = 0
    for path, n in zip(paths, nstripes_per_file):
        for stripe in (
            range(n)
            if filters is None
            else cudf.io.orc._filter_stripes(filters, path)
        ):
            if (
                not filtering_columns_first
                or not is_filtered_partition_empty[partition_idx]
            ):
                dsk[(name, N)] = (
                    _read_orc_stripe,
                    fs,
                    path,
                    stripe,
                    columns,
                    kwargs,
                )
                N += 1
            partition_idx += 1

    divisions = [None] * (len(dsk) + 1)
    res = dd.core.new_dd_object(dsk, name, meta, divisions)

    if not filtering_columns_first:
        return res
    else:
        reordering_columns_required = (
            columns_in_predicate + columns != all_columns
        )

        # Create a delayed function to horizontally concatenate filtered
        # filtering column partitions with partitions of the remaining
        # columns for which we only read in relevant stripes.
        if reordering_columns_required:

            @delayed(pure=True)
            def _hcat(a, b):
                return cudf.concat([a, b], axis=1)[all_columns]

        else:

            @delayed(pure=True)
            def _hcat(a, b):
                return cudf.concat([a, b], axis=1)

        # Return a data frame comprised of delayed horizontal concatenation.
        remaining_df_partitions = res.to_delayed()
        concatenated_meta = cudf.concat([filtered_df._meta, res._meta], axis=1)
        return dd.from_delayed(
            [
                _hcat(filtered, remaining)
                for filtered, remaining in zip(
                    filtered_df_partitions, remaining_df_partitions
                )
            ],
            concatenated_meta[all_columns]
            if reordering_columns_required
            else concatenated_meta,
        )


def write_orc_partition(
    df,
    path,
    fs,
    filename,
    compression,
    stripe_size_bytes,
    stripe_size_rows,
    row_index_stride,
):
    full_path = fs.sep.join([path, filename])
    with fs.open(full_path, mode="wb") as out_file:
        if not isinstance(out_file, IOBase):
            out_file = BufferedWriter(out_file)
        cudf.io.to_orc(
            df,
            out_file,
            compression=compression,
            stripe_size_bytes=stripe_size_bytes,
            stripe_size_rows=stripe_size_rows,
            row_index_stride=row_index_stride,
        )
    return full_path


def to_orc(
    df,
    path,
    write_index=True,
    storage_options=None,
    compression=None,
    stripe_size_bytes=None,
    stripe_size_rows=None,
    row_index_stride=None,
    compute=True,
    **kwargs,
):
    """Write a dask_cudf dataframe to ORC file(s) (one file per partition).

    Parameters
    ----------
    df : dask_cudf.DataFrame
    path: string or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, optional
        Whether or not to write the index. Defaults to True.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    compression : string or dict, optional
    stripe_size_bytes: integer or None, default None
        Maximum size of each stripe of the output.
        If None, 67108864 (64MB) will be used.
    stripe_size_rows: integer or None, default None 1000000
        Maximum number of rows of each stripe of the output.
        If None, 1000000 will be used.
    row_index_stride: integer or None, default None 10000
        Row index stride (maximum number of rows in each row group).
        If None, 10000 will be used.
    compute : bool, optional
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    """

    from dask import compute as dask_compute, delayed

    # TODO: Use upstream dask implementation once available
    #       (see: Dask Issue#5596)

    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, _ = get_fs_token_paths(
        path, mode="wb", storage_options=storage_options
    )
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)

    if write_index:
        df = df.reset_index()
    else:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)

    fs.mkdirs(path, exist_ok=True)

    # Use i_offset and df.npartitions to define file-name list
    filenames = ["part.%i.orc" % i for i in range(df.npartitions)]

    # write parts
    dwrite = delayed(write_orc_partition)
    parts = [
        dwrite(
            d,
            path,
            fs,
            filename,
            compression,
            stripe_size_bytes,
            stripe_size_rows,
            row_index_stride,
        )
        for d, filename in zip(df.to_delayed(), filenames)
    ]

    if compute:
        return dask_compute(*parts)

    return delayed(list)(parts)
