# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import copy
from collections import defaultdict
from contextlib import ExitStack
from io import BufferedWriter, IOBase

import pandas as pd
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from pyarrow import orc as orc

from dask import dataframe as dd
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.core import DataFrame, Scalar, new_dd_object
from dask.dataframe.io.parquet.core import apply_filters
from dask.dataframe.io.parquet.utils import _flatten_filters
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import apply, natural_sort_key

import cudf
from cudf.core.column import as_column, build_categorical_column

#
#  CudfORCEngine
#  (Used by the EXPERIMENTAL read_orc Implementation)
#


class CudfORCEngine:
    @classmethod
    def get_dataset_info(
        cls,
        path,
        columns=None,
        index=None,
        filters=None,
        gather_statistics=True,
        dataset_kwargs=None,
        storage_options=None,
    ):

        # Process file path(s)
        fs, _, paths = get_fs_token_paths(
            path, mode="rb", storage_options=storage_options or {}
        )

        # Generate a full list of files and
        # directory partitions. We would like to use
        # something like the pyarrow.dataset API to
        # do this, but we will need to do it manually
        # until ORC is supported upstream.
        directory_partitions = []
        directory_partition_keys = {}
        if len(paths) == 1 and not fs.isfile(paths[0]):
            root_dir = paths[0]
            paths = collect_files(root_dir, fs)
            (
                paths,
                directory_partitions,
                directory_partition_keys,
            ) = collect_partitions(paths, root_dir, fs)

        # Sample the 0th file to geth the schema
        with fs.open(paths[0], "rb") as f:
            o = orc.ORCFile(f)
            schema = o.schema

        # Save a list of directory-partition columns and a list
        # of file columns that we will need statistics for
        dir_columns_need_stats = {
            col
            for col in _flatten_filters(filters)
            if col in directory_partition_keys
        } | ({index} if index in directory_partition_keys else set())
        file_columns_need_stats = {
            col for col in _flatten_filters(filters) if col in schema.names
        }
        # Before including the index column, raise an error
        # if the user is trying to filter with gather_statistics=False
        if file_columns_need_stats and gather_statistics is False:
            raise ValueError(
                "Cannot filter ORC stripes when `gather_statistics=False`."
            )
        file_columns_need_stats |= {index} if index in schema.names else set()

        # Convert the orc schema to a pyarrow schema
        # and check that the columns agree with the schema
        pa_schema = _get_pyarrow_dtypes(schema, categories=None)
        if columns is not None:
            ex = set(columns) - (
                set(pa_schema) | set(directory_partition_keys)
            )
            if ex:
                raise ValueError(
                    "Requested columns (%s) not in schema (%s)"
                    % (ex, set(schema))
                )

        # Return a final `dataset_info` dictionary.
        # We use a dictionary here to make the `ORCEngine`
        # API as flexible as possible.
        return {
            "fs": fs,
            "paths": paths,
            "orc_schema": schema,
            "pa_schema": pa_schema,
            "dir_columns_need_stats": dir_columns_need_stats,
            "file_columns_need_stats": file_columns_need_stats,
            "directory_partitions": directory_partitions,
            "directory_partition_keys": directory_partition_keys,
        }

    @classmethod
    def construct_output_meta(
        cls,
        dataset_info,
        index=None,
        columns=None,
        sample_data=True,
        read_kwargs=None,
    ):

        # Use dataset_info to define `columns`
        schema = dataset_info["pa_schema"]
        directory_partition_keys = dataset_info["directory_partition_keys"]
        columns = list(schema) if columns is None else columns

        # Construct initial meta
        pd_meta = _meta_from_dtypes(columns, schema, None, [])

        # Deal with hive-partitioned data
        for column, uniques in (directory_partition_keys or {}).items():
            if column not in pd_meta.columns:
                pd_meta[column] = pd.Series(
                    pd.Categorical(categories=uniques, values=[]),
                    index=pd_meta.index,
                )

        # Set index if one was specified
        if index:
            pd_meta.set_index(index, inplace=True)

        # Return direct conversion if sample_data=False
        if sample_data is False:
            return cudf.from_pandas(pd_meta)

        # Read 0th stripe to get cudf metadata
        fs = dataset_info["fs"]
        path = dataset_info["paths"][0]
        with fs.open(path, "rb") as f:
            o = orc.ORCFile(f)
            stripes = [0] if o.nstripes else None
            f.seek(0)
            cudf_meta = cudf.read_orc(
                f, stripes=stripes, columns=columns, **(read_kwargs or {}),
            ).iloc[:0]

        # Use pandas metadata to update missing
        # columns in cudf_meta (directory partitions)
        for col in pd_meta.columns:
            if col not in cudf_meta.columns:
                cudf_meta[col] = as_column(pd_meta[col])

        return cudf_meta

    @classmethod
    def construct_partition_plan(
        cls,
        meta,
        dataset_info,
        filters=None,
        split_stripes=True,
        aggregate_files=False,
        gather_statistics=True,
        allow_worker_gather=None,
    ):

        # Extract column and index from meta
        columns = list(meta.columns)
        index = meta.index.name

        # Extract necessary dataset_info values
        directory_partition_keys = dataset_info["directory_partition_keys"]

        # Set the file-aggregation depth if the data has
        # directory partitions, and one of these partition
        # columns was specified by `aggregate_files`
        dir_agg_depth = 0
        if isinstance(aggregate_files, str):
            try:
                dir_agg_depth = (
                    list(directory_partition_keys).index(aggregate_files) + 1
                )
            except ValueError:
                raise ValueError(
                    f"{aggregate_files} is not a recognized partition column. "
                    f"Please check the aggregate_files argument."
                )

        # Gather a list of partitions and corresponding
        # statistics.  Each element in this initial partition
        # list will only correspond to a single path. The
        # following `aggregate_files` method is required
        # to coalesce multiple paths into a single
        # `read_partition` task. Note that `_gather_parts`
        # will use `cls.filter_file_stripes` to apply filters
        # on each path (and collect statistics) independently.
        # Therefore, this call can be parallelized over the paths.
        npaths = len(dataset_info["paths"])
        worker_gather = (allow_worker_gather is True) or (
            npaths > 1
            and (split_stripes or filters or index)
            and allow_worker_gather is not False
            and not aggregate_files  # Worker-gather can change file agg
        )
        if worker_gather:
            # Collect partition plan on workers (in parallel)
            gather_parts_dsk = {}
            name = "gather-orc-parts-" + tokenize(
                meta,
                dataset_info,
                filters,
                split_stripes,
                aggregate_files,
                gather_statistics,
            )
            finalize_list = []
            for i in range(npaths):
                finalize_list.append((name, i))
                gather_parts_dsk[finalize_list[-1]] = (
                    apply,
                    cls._gather_parts,
                    [dataset_info],
                    {
                        "path_indices": [i],
                        "index": index,
                        "columns": columns,
                        "filters": filters,
                        "split_stripes": split_stripes,
                        "aggregate_files": aggregate_files,
                        "gather_statistics": gather_statistics,
                        "dir_agg_depth": dir_agg_depth,
                    },
                )

            def _combine_parts(parts_and_stats):
                parts, stats = [], []
                for part, stat in parts_and_stats:
                    parts += part
                    stats += stat
                return parts, stats

            gather_parts_dsk["final-" + name] = (_combine_parts, finalize_list)
            parts, statistics = Delayed(
                "final-" + name, gather_parts_dsk
            ).compute()
        else:
            # Collect partition plan on client (serial)
            parts, statistics = cls._gather_parts(
                dataset_info,
                index=index,
                columns=columns,
                filters=filters,
                split_stripes=split_stripes,
                aggregate_files=aggregate_files,
                gather_statistics=gather_statistics,
                dir_agg_depth=dir_agg_depth,
            )

        # Use avilable statistics to calculate divisions
        divisions = None
        if index and statistics:
            divisions = cls._calculate_divisions(index, statistics)

        # Aggregate adjacent partitions together
        # (when possible/desired)
        if aggregate_files:
            parts, divisions = cls._aggregate_files(
                parts,
                dir_agg_depth=dir_agg_depth,
                split_stripes=split_stripes,
                statistics=statistics,
                divisions=divisions,
            )

        # Define common kwargs
        common_kwargs = {
            "fs": dataset_info["fs"],
            "schema": dataset_info["pa_schema"],
            "partition_uniques": dataset_info["directory_partition_keys"],
            "filters": filters,
        }

        return parts, divisions, common_kwargs

    @classmethod
    def _gather_parts(
        cls,
        dataset_info,
        path_indices=None,
        index=None,
        columns=None,
        filters=None,
        split_stripes=True,
        aggregate_files=False,
        gather_statistics=True,
        dir_agg_depth=0,
    ):
        """Gather partitioning plan for every path in the dataset"""

        # Extract necessary info from dataset_info
        fs = dataset_info["fs"]
        paths = dataset_info["paths"]
        schema = dataset_info["orc_schema"]
        directory_partitions = dataset_info["directory_partitions"]
        dir_columns_need_stats = dataset_info["dir_columns_need_stats"]
        file_columns_need_stats = dataset_info["file_columns_need_stats"]

        # Assume we are processing all paths if paths=None
        if path_indices is None:
            path_indices = range(len(paths))

        # Main loop(s) to gather stripes/statistics for
        # each file. After this, each element of `parts` will
        # correspond to a group of stripes for a single file/path.
        parts = []
        statistics = []
        offset = 0
        for i in path_indices:
            path = paths[i]
            hive_part = directory_partitions[i] if directory_partitions else []
            hive_part_need_stats = [
                (k, v) for k, v in hive_part if k in dir_columns_need_stats
            ]
            if split_stripes:
                with fs.open(path, "rb") as f:
                    o = orc.ORCFile(f)
                    nstripes = o.nstripes
                    if schema != o.schema:
                        raise ValueError(
                            "Incompatible schemas while parsing ORC files"
                        )
                    stripes, stats = cls.filter_file_stripes(
                        fs=fs,
                        orc_file=o,
                        filters=filters,
                        stat_columns=file_columns_need_stats,
                        stat_hive_part=hive_part_need_stats,
                        file_handle=f,
                        file_path=path,
                        gather_statistics=gather_statistics,
                    )
                    if stripes == []:
                        continue
                    if offset:
                        new_part_stripes = stripes[0:offset]
                        if new_part_stripes:
                            parts.append([(path, new_part_stripes, hive_part)])
                            if gather_statistics:
                                statistics += cls._aggregate_stats(
                                    stats[0:offset]
                                )
                    while offset < nstripes:
                        new_part_stripes = stripes[
                            offset : offset + int(split_stripes)
                        ]
                        if new_part_stripes:
                            parts.append([(path, new_part_stripes, hive_part)])
                            if gather_statistics:
                                statistics += cls._aggregate_stats(
                                    stats[offset : offset + int(split_stripes)]
                                )
                        offset += int(split_stripes)
                    if (
                        aggregate_files
                        and int(split_stripes) > 1
                        and dir_agg_depth < 1
                    ):
                        offset -= nstripes
                    else:
                        offset = 0
            else:
                stripes, stats = cls.filter_file_stripes(
                    fs=fs,
                    orc_file=None,
                    filters=filters,
                    stat_columns=file_columns_need_stats,
                    stat_hive_part=hive_part_need_stats,
                    file_path=path,
                    file_handle=None,
                    gather_statistics=gather_statistics,
                )
                if stripes == []:
                    continue
                parts.append([(path, stripes, hive_part)])
                if gather_statistics:
                    statistics += cls._aggregate_stats(stats)

        return parts, statistics

    @classmethod
    def filter_file_stripes(
        cls,
        orc_file=None,
        filters=None,
        stat_hive_part=None,
        file_path=None,
        fs=None,
        stat_columns=None,
        file_handle=None,
        gather_statistics=True,
    ):

        # Use cudf to gather stripe statistics
        cudf_statistics = []
        if gather_statistics:
            if file_handle is None:
                with fs.open(file_path, "rb") as f:
                    f.seek(0)
                    cudf_statistics = cudf.io.orc.read_orc_statistics(
                        [f], columns=set(stat_columns) | {"col0"},
                    )[1]
            else:
                file_handle.seek(0)
                cudf_statistics = cudf.io.orc.read_orc_statistics(
                    [file_handle], columns=set(stat_columns) | {"col0"},
                )[1]
            stripes = list(range(len(cudf_statistics)))

        # Populate statistics with stripe statistics from cudf
        # and "known" statistics from the directory partitioning
        statistics = []
        for i, stripe in enumerate(stripes):
            stats = {"file-path": file_path}
            column_stats = []
            columns_populated = []
            if cudf_statistics:
                for k, v in cudf_statistics[i].items():
                    if k == "col0":
                        stats["num-rows"] = v.get("number_of_values", None)
                    elif k in stat_columns:
                        columns_populated.append(k)
                        column_stats.append(
                            {
                                "name": k,
                                "count": v.get("number_of_values", None),
                                "min": v.get("minimum", None),
                                "max": v.get("maximum", None),
                            }
                        )
            for (k, v) in stat_hive_part:
                if k not in columns_populated:
                    column_stats.append(
                        {
                            "name": k,
                            "count": stats["num-rows"],
                            "min": v,
                            "max": v,
                        }
                    )
            stats["columns"] = column_stats
            statistics.append(stats)

        # Apply filters (if necessary)
        if filters:
            stripes, statistics = apply_filters(stripes, statistics, filters)

        return stripes, statistics

    @classmethod
    def _calculate_divisions(cls, index, statistics):
        """Use statistics to calculate divisions"""
        if statistics:
            divisions = []
            for icol, column_stats in enumerate(
                statistics[0].get("columns", [])
            ):
                if column_stats.get("name", None) == index:
                    divisions = [
                        column_stats.get("min", None),
                        column_stats.get("max", None),
                    ]
                    break
            if divisions and None not in divisions:
                for stat in statistics[1:]:
                    next_division = stat["columns"][icol].get("max", None)
                    if next_division is None or next_division < divisions[-1]:
                        return None
                    divisions.append(next_division)
            return divisions
        return None

    @classmethod
    def _aggregate_stats(cls, statistics):
        """Aggregate a list of statistics"""

        if statistics:

            # Check if we are already "aggregated"
            nstats = len(statistics)
            if nstats == 1:
                return statistics

            # Populate statistic lists
            counts = []
            column_counts = defaultdict(list)
            column_mins = defaultdict(list)
            column_maxs = defaultdict(list)
            use_count = statistics[0].get("num-rows", None) is not None
            for stat in statistics:
                if use_count:
                    counts.append(stat.get("num-rows"))
                for col_stats in stat["columns"]:
                    name = col_stats["name"]
                    if use_count:
                        column_counts[name].append(col_stats.get("count"))
                    column_mins[name].append(col_stats.get("min", None))
                    column_maxs[name].append(col_stats.get("max", None))

            # Perform aggregation
            output = {}
            output["file-path"] = statistics[0].get("file-path", None)
            if use_count:
                output["row-count"] = sum(counts)
            column_stats = []
            for k in column_counts.keys():
                column_stat = {"name": k}
                if use_count:
                    column_stat["count"] = sum(column_counts[k])
                try:
                    column_stat["min"] = min(column_mins[k])
                    column_stat["max"] = max(column_maxs[k])
                except TypeError:
                    column_stat["min"] = None
                    column_stat["max"] = None
                column_stats.append(column_stat)
            output["columns"] = column_stats
            return output
        else:
            return {}

    @classmethod
    def _aggregate_files(
        cls,
        parts,
        dir_agg_depth=0,
        split_stripes=1,
        divisions=None,
        statistics=None,  # Not used (yet)
    ):
        if int(split_stripes) > 1 and len(parts) > 1:
            new_parts = []
            new_divisions = [divisions[0]] if divisions else None
            new_max = divisions[1] if divisions else None
            new_part = parts[0]
            nstripes = len(new_part[0][1])
            hive_parts = new_part[0][2]
            for i, part in enumerate(parts[1:]):
                next_nstripes = len(part[0][1])
                new_hive_parts = part[0][2]
                # For partitioned data, we do not allow file
                # aggregation between different hive partitions
                if (next_nstripes + nstripes <= split_stripes) and (
                    hive_parts[:dir_agg_depth]
                    == new_hive_parts[:dir_agg_depth]
                ):
                    new_part.append(part[0])
                    new_max = divisions[i] if divisions else None
                    nstripes += next_nstripes
                else:
                    new_parts.append(new_part)
                    if divisions:
                        new_divisions.append(new_max)
                    new_part = part
                    new_max = divisions[i] if divisions else None
                    nstripes = next_nstripes
                    hive_parts = new_hive_parts
            new_parts.append(new_part)
            if divisions:
                new_divisions.append(new_max)
            return new_parts, new_divisions
        else:
            return parts, divisions

    @classmethod
    def read_partition(
        cls,
        parts,
        columns,
        fs=None,
        filters=None,
        schema=None,
        partition_uniques=None,
        **kwargs,
    ):
        # Create a seperate dataframe for each directory partition.
        # We are only creating a single cudf dataframe if there
        # are no partition columns.
        partitions = []
        partition_uniques = partition_uniques or {}
        if columns:
            # Separate file columns and partition columns
            file_columns = [c for c in columns if c in set(schema)]
            partition_columns = [c for c in columns if c not in set(schema)]
        else:
            file_columns, partition_columns = None, list(partition_uniques)

        dfs = []
        path, stripes, hive_parts = parts[0]
        path_list = [path]
        stripe_list = [stripes]
        for path, stripes, next_hive_parts in parts[1:]:
            if hive_parts == next_hive_parts:
                path_list.append(path)
                stripe_list.append(stripes)
            else:
                dfs.append(
                    cls._read_partition(
                        fs, path_list, file_columns, stripe_list, **kwargs,
                    )
                )
                partitions.append(hive_parts)
                path_list = [path]
                stripe_list = [stripes]
                hive_parts = next_hive_parts
        dfs.append(
            cls._read_partition(
                fs, path_list, file_columns, stripe_list, **kwargs,
            )
        )
        partitions.append(hive_parts)

        # Add partition columns to each partition dataframe
        for i, hive_parts in enumerate(partitions):
            for (part_name, cat) in hive_parts:
                if part_name in partition_columns:
                    # We read from file paths, so the partition
                    # columns are NOT in our table yet.
                    categories = partition_uniques[part_name]

                    col = (
                        as_column(categories.index(cat))
                        .as_frame()
                        .repeat(len(dfs[i]))
                        ._data[None]
                    )
                    dfs[i][part_name] = build_categorical_column(
                        categories=categories,
                        codes=as_column(col.base_data, dtype=col.dtype),
                        size=col.size,
                        offset=col.offset,
                        ordered=False,
                    )
        return cudf.concat(dfs)

    @classmethod
    def _read_partition(cls, fs, path_list, columns, stripe_list, **kwargs):

        with ExitStack() as stack:
            if cudf.utils.ioutils._is_local_filesystem(fs):
                # Let cudf open the files if this is
                # a local file system
                _source_list = path_list
            else:
                # Use fs.open to pass file handles to cudf
                _source_list = [
                    stack.enter_context(fs.open(path, "rb"))
                    for path in path_list
                ]
            df = cudf.io.read_orc(
                _source_list, columns=columns, stripes=stripe_list, **kwargs,
            )
        return df


#
#  EXPERIMENTAL read_orc Implementation
#


def read_orc(
    path,
    engine="cudf",
    legacy=False,
    columns=None,
    index=None,
    filters=None,
    split_stripes=1,
    aggregate_files=None,
    storage_options=None,
    gather_statistics=True,
    allow_worker_gather=None,
    sample_data=None,
    read_kwargs=None,
):
    """Read dataframe from ORC file(s)

    Parameters
    ----------
    path: str or list(str)
        Location of file(s), which can be a full URL with protocol
        specifier, and may include glob character if a single string.
    engine: str, default "cudf"
        IO engine label to pass to cudf backend.
    legacy: bool, default False
        Whether to use the legacy ``dask_cudf.read_orc`` implementation.
    columns: None or list(str)
        Columns to load. If None, loads all.
    index: str
        Column name to set as index.
    filters : None or list of tuple or list of lists of tuples
        Specifies a filter predicate used to filter out row groups using
        statistics stored for each row group as Parquet metadata. Row
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
    split_stripes: int or False
        Maximum number of ORC stripes to include in each output-DataFrame
        partition. Use False to specify a 1-to-1 mapping between files
        and partitions. Default is 1.
    aggregate_files : bool or str, default False
        Whether distinct file paths may be aggregated into the same output
        partition. A setting of True means that any two file paths may be
        aggregated into the same output partition, while False means that
        inter-file aggregation is prohibited. If the name of a partition
        column is specified, any file within the same partition directory
        (e.g. ``"/<aggregate_files>=*/"``) may be aggregated.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    gather_statistics : bool, default True
        Whether to gather file and stripe statistics from the orc metadata.
    allow_worker_gather : bool, optional
        Whether to parallelize the gathering and processing of orc metadata.
    sample_data : bool, optional
        Whether to sample data to construct output collection metadata.
    read_kwargs : dict, optional
        Dictionary of key-word arguments to pass to the partition-level
        IO function.

    Returns
    -------
    dask_cudf.DataFrame (even if there is only one column)
    """

    # Check if we are using the legacy engine
    if legacy:
        return read_orc_legacy(
            path,
            engine=engine,
            columns=columns,
            filters=filters,
            storage_options=storage_options,
            **(read_kwargs or {}),
        )

    # Add `engine` to `read_kwargs`
    read_kwargs = read_kwargs or {}
    read_kwargs["engine"] = engine

    # Check that `index` is legal
    if isinstance(index, list):
        if len(index) > 1:
            raise ValueError(
                f"index={index} not supported. "
                f"Please use a single column name."
            )
        index = index[0]

    # Set engine to CudfORCEngine
    engine = CudfORCEngine

    # Let engine convert the paths into a dictionary
    # of engine-specific datset information
    dataset_info = engine.get_dataset_info(
        path,
        columns=columns,
        index=index,
        filters=filters,
        gather_statistics=gather_statistics,
        storage_options=storage_options,
    )

    # Construct the `_meta` for the output collection.
    # Note that we do this before actually generating
    # the "plan" for output partitions.
    meta = engine.construct_output_meta(
        dataset_info,
        index=index,
        columns=columns,
        sample_data=sample_data,
        read_kwargs=read_kwargs,
    )

    # Construct the output-partition "plan"
    parts, divisions, common_kwargs = engine.construct_partition_plan(
        meta,
        dataset_info,
        filters=filters,
        split_stripes=split_stripes,
        aggregate_files=aggregate_files,
        gather_statistics=gather_statistics,
        allow_worker_gather=allow_worker_gather,
    )

    # Add read_kwargs to common_kwargs
    common_kwargs.update(read_kwargs or {})

    # Construct and return a Blockwise layer
    label = "read-orc-"
    output_name = label + tokenize(
        dataset_info,
        columns,
        index,
        split_stripes,
        aggregate_files,
        filters,
        read_kwargs,
    )
    layer = DataFrameIOLayer(
        output_name,
        columns,
        parts,
        ORCFunctionWrapper(
            columns, engine, index, common_kwargs=common_kwargs
        ),
        label=label,
    )
    graph = HighLevelGraph({output_name: layer}, {output_name: set()})
    divisions = divisions or ([None] * (len(parts) + 1))
    return new_dd_object(graph, output_name, meta, divisions)


#
#  EXPERIMENTAL read_orc Utilities
#


class ORCFunctionWrapper:
    """
    ORC Function-Wrapper Class
    Reads ORC data from disk to produce a partition.
    """

    def __init__(self, columns, engine, index, common_kwargs=None):
        self.columns = columns
        self.engine = engine
        self.index = index
        self.common_kwargs = common_kwargs or {}

    def project_columns(self, columns):
        """Return a new ORCFunctionWrapper object with
        a sub-column projection.
        """
        if columns == self.columns:
            return self
        func = copy.deepcopy(self)
        func.columns = columns
        return func

    def __call__(self, parts):
        _df = self.engine.read_partition(
            parts, self.columns, **self.common_kwargs,
        )
        if self.index:
            _df.set_index(self.index, inplace=True)
        return _df


def _is_data_file_path(path, fs, ignore_prefix=None, require_suffix=None):
    # Private utility to check if a path is a data file

    # Check that we are not ignoring this path/dir
    if ignore_prefix and path.startswith(ignore_prefix):
        return False
    # If file, check that we are allowing this suffix
    if fs.isfile(path) and require_suffix and path.endswith(require_suffix):
        return False
    return True


def collect_files(root, fs, ignore_prefix="_", require_suffix=None):
    # Utility to recursively collect all files within
    # a root directory.

    # First, check if we are dealing with a file
    if fs.isfile(root):
        if _is_data_file_path(
            root,
            fs,
            ignore_prefix=ignore_prefix,
            require_suffix=require_suffix,
        ):
            return [root]
        return []

    # Otherwise, recursively handle each item in
    # the current `root` directory
    all_paths = []
    for sub in fs.ls(root):
        all_paths += collect_files(
            sub,
            fs,
            ignore_prefix=ignore_prefix,
            require_suffix=require_suffix,
        )

    return all_paths


def collect_partitions(file_list, root, fs, partition_sep="=", dtypes=None):
    # Utility to manually collect hive-style
    # directory-partitioning information from a dataset.

    # Always sort files by `natural_sort_key` to ensure
    # files within the same directory partition are together
    files = sorted(file_list, key=natural_sort_key)

    # Construct partitions
    parts = []
    root_len = len(root.split(fs.sep))
    dtypes = dtypes or {}
    unique_parts = defaultdict(set)
    for path in files:
        # Strip root and file name
        _path = path.split(fs.sep)[root_len:-1]
        partition = []
        for d in _path:
            _split = d.split(partition_sep)
            if len(_split) == 2:
                col = _split[0]
                # Interpret partition key as int, float, or str
                raw_parition_key = (
                    dtypes[col](_split[1]) if col in dtypes else _split[1]
                )
                try:
                    parition_key = int(raw_parition_key)
                except ValueError:
                    try:
                        parition_key = float(raw_parition_key)
                    except ValueError:
                        parition_key = raw_parition_key
                # Append partition name-key tuple to `partition` list
                partition.append((_split[0], parition_key,))
        if partition:
            for (k, v) in partition:
                unique_parts[k].add(v)
            parts.append(partition)

    return files, parts, {k: list(v) for k, v in unique_parts.items()}


#
#  LEGACY read_orc Implementation
#


def read_orc_legacy(
    path, columns=None, filters=None, storage_options=None, **kwargs
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
    for path, n in zip(paths, nstripes_per_file):
        for stripe in (
            range(n)
            if filters is None
            else cudf.io.orc._filter_stripes(filters, path)
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

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def _read_orc_stripe(fs, path, stripe, columns, kwargs=None):
    """Pull out specific columns from specific stripe"""
    if kwargs is None:
        kwargs = {}
    with fs.open(path, "rb") as f:
        df_stripe = cudf.read_orc(
            f, stripes=[stripe], columns=columns, **kwargs
        )
    return df_stripe


#
#  to_orc Implementation
#


def to_orc(
    df,
    path,
    write_index=True,
    append=False,
    partition_on=None,
    storage_options=None,
    compute=True,
    compute_kwargs=None,
    **kwargs,
):
    """Write a dask_cudf dataframe to ORC file(s). There will be one output
    file per partition, unless `partition_on` is specified (in which case
    each partition may be split into multiple files).

    Parameters
    ----------
    df : dask_cudf.DataFrame
    path: string or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, optional
        Whether or not to write the index. Defaults to True.
    append : boolean, default False
        If False (default), construct data-set from scratch. If True, add new
        file(s) to an existing data-set (if one exists).
    partition_on : list, default None
        Construct directory-based partitioning by splitting on these fields'
        values. Each dask partition will result in one or more datafiles,
        there will be no global groupby.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    compute : bool, optional
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    compute_kwargs : dict, default True
        Options to be passed in to the compute method
    **kwargs :
        Key-word arguments to be passed on to ``cudf.to_orc``.
    """

    # TODO: Use upstream dask implementation once available
    #       (see: Dask Issue#5596)

    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, _ = get_fs_token_paths(
        path, mode="wb", storage_options=storage_options
    )
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)

    # Reset index if we are writing it, otherwise
    # we might as well drop it
    df = df.reset_index(drop=not write_index)

    # Use i_offset and df.npartitions to define file-name list.
    # For `append=True`, we use the total number of existing files
    # to define `i_offset`
    fs.mkdirs(path, exist_ok=True)
    i_offset = len(collect_files(path, fs)) if append else 0
    filenames = [f"part.{i+i_offset}.orc" for i in range(df.npartitions)]

    # Construct IO graph
    dsk = {}
    name = "to-orc-" + tokenize(
        df,
        fs,
        path,
        write_index,
        partition_on,
        i_offset,
        storage_options,
        kwargs,
    )
    final_name = name + "-final"
    for d, filename in enumerate(filenames):
        dsk[(name, d)] = (
            apply,
            write_orc_partition,
            [(df._name, d), path, fs, filename, partition_on, kwargs],
        )
    part_tasks = list(dsk.keys())
    dsk[(final_name, 0)] = (lambda x: None, part_tasks)
    graph = HighLevelGraph.from_collections(
        (final_name, 0), dsk, dependencies=[df]
    )

    # Compute or return future
    if compute:
        if compute_kwargs is None:
            compute_kwargs = dict()
        return compute_as_if_collection(
            DataFrame, graph, part_tasks, **compute_kwargs
        )
    return Scalar(graph, final_name, "")


def write_orc_partition(df, path, fs, filename, partition_on, kwargs):
    if partition_on:
        cudf.io.to_orc(
            df,
            path,
            fs=fs,
            partition_cols=partition_on,
            partition_file_name=filename,
            **kwargs,
        )
    else:
        full_path = fs.sep.join([path, filename])
        with fs.open(full_path, mode="wb") as out_file:
            if not isinstance(out_file, IOBase):
                out_file = BufferedWriter(out_file)
            cudf.io.to_orc(df, out_file, **kwargs)
        return full_path
