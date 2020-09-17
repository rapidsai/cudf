# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf.core import column
import functools
import warnings
import datetime

import pyarrow as pa
from pyarrow import orc as orc

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils
from cudf.utils import filterutils

import cudf.utils.metadata.orc_column_statistics_pb2 as cs_pb2


def _make_empty_df(filepath_or_buffer, columns):
    orc_file = orc.ORCFile(filepath_or_buffer)
    schema = orc_file.schema
    col_names = schema.names if columns is None else columns
    empty = {}
    for col_name in col_names:
        empty[col_name] = cudf.Series(
            [], dtype=schema.field(col_name).type.to_pandas_dtype(),
        )
    return cudf.DataFrame(empty)


def _parse_column_statistics(cs, column_statistics_blob):
    # Initialize stats to return and parse stats blob
    column_statistics = {}
    cs.ParseFromString(column_statistics_blob)

    # Load from parsed stats blob into stats to return
    if not cs.HasField("numberOfValues") or not cs.HasField("hasNull"):
        return None
    column_statistics["number_of_values"] = cs.numberOfValues
    column_statistics["has_null"] = cs.hasNull
    if cs.HasField("intStatistics"):
        column_statistics["minimum"] = cs.intStatistics.minimum
        column_statistics["maximum"] = cs.intStatistics.maximum
        column_statistics["sum"] = cs.intStatistics.sum
    elif cs.HasField("doubleStatistics"):
        column_statistics["minimum"] = cs.doubleStatistics.minimum
        column_statistics["maximum"] = cs.doubleStatistics.maximum
        column_statistics["sum"] = cs.doubleStatistics.sum
    elif cs.HasField("stringStatistics"):
        column_statistics["minimum"] = cs.stringStatistics.minimum
        column_statistics["maximum"] = cs.stringStatistics.maximum
        column_statistics["sum"] = cs.stringStatistics.sum
    elif cs.HasField("bucketStatistics"):
        column_statistics["true_count"] = cs.bucketStatistics.count[0]
        column_statistics["false_count"] = (
            column_statistics["number_of_values"]
            - column_statistics["true_count"]
        )
    elif cs.HasField("decimalStatistics"):
        column_statistics["minimum"] = cs.decimalStatistics.minimum
        column_statistics["maximum"] = cs.decimalStatistics.maximum
        column_statistics["sum"] = cs.decimalStatistics.sum
    elif cs.HasField("dateStatistics"):
        column_statistics["minimum"] = datetime.datetime.fromtimestamp(
            datetime.timedelta(cs.dateStatistics.minimum).total_seconds(),
            datetime.timezone.utc,
        )
        column_statistics["maximum"] = datetime.datetime.fromtimestamp(
            datetime.timedelta(cs.dateStatistics.maximum).total_seconds(),
            datetime.timezone.utc,
        )
    elif cs.HasField("timestampStatistics"):
        # Before ORC-135, the local timezone offset was included and they were
        # stored as minimum and maximum. After ORC-135, the timestamp is
        # adjusted to UTC before being converted to milliseconds and stored
        # in minimumUtc and maximumUtc.
        # TODO: Support minimum and maximum by reading writer's local timezone
        if cs.timestampStatistics.HasField(
            "minimumUtc"
        ) and cs.timestampStatistics.HasField("maximumUtc"):
            column_statistics["minimum"] = datetime.datetime.fromtimestamp(
                cs.timestampStatistics.minimumUtc / 1000, datetime.timezone.utc
            )
            column_statistics["maximum"] = datetime.datetime.fromtimestamp(
                cs.timestampStatistics.maximumUtc / 1000, datetime.timezone.utc
            )
    elif cs.HasField("binaryStatistics"):
        column_statistics["sum"] = cs.binaryStatistics.sum

    return column_statistics


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc_statistics()
@functools.lru_cache(maxsize=8)
def read_orc_statistics_with_cache(
    filepath_or_buffer, columns=None, **kwargs,
):
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        path_or_data=filepath_or_buffer, compression=None, **kwargs
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    # Read in statistics and unpack
    statistics = libcudf.orc.read_orc_statistics(filepath_or_buffer)
    if not statistics:
        return None
    (column_names, raw_file_statistics, *raw_stripes_statistics,) = statistics

    # Parse column names
    column_names = [
        column_name.decode("utf-8") for column_name in column_names
    ]

    # Parse statistics
    cs = cs_pb2.ColumnStatistics()
    file_statistics = {}
    stripes_statistics = []
    for i, raw_file_stats in enumerate(raw_file_statistics):
        if columns is None or column_names[i] in columns:
            parsed_statistics = _parse_column_statistics(cs, raw_file_stats)
            if not parsed_statistics:
                return None
            file_statistics[column_names[i]] = parsed_statistics
    for raw_stripe_statistics in raw_stripes_statistics:
        stripe_statistics = {}
        for i, raw_file_stats in enumerate(raw_stripe_statistics):
            if columns is None or column_names[i] in columns:
                parsed_statistics = _parse_column_statistics(
                    cs, raw_file_stats
                )
                if not parsed_statistics:
                    return None
                stripe_statistics[column_names[i]] = parsed_statistics
        stripes_statistics.append(stripe_statistics)

    return file_statistics, stripes_statistics


def read_orc_statistics(
    filepath_or_buffer, columns=None, **kwargs,
):
    return read_orc_statistics_with_cache(
        filepath_or_buffer,
        tuple(columns) if columns is not None else None,
        **kwargs,
    )


def _filter_stripes(
    filters,
    filepath_or_buffer,
    stripes=None,
    skip_rows=None,
    num_rows=None,
    already_prepared=False,
):
    if not already_prepared:
        filters = filterutils._prepare_filters(filters)

    # Get columns relevant to filtering
    columns_in_predicate = [
        col for conjunction in filters for (col, op, val) in conjunction
    ]

    # Read and parse file-level and stripe-level statistics
    file_statistics, stripes_statistics = read_orc_statistics(
        filepath_or_buffer, columns_in_predicate
    )

    # Filter using file-level statistics
    if not filterutils._apply_filters(filters, file_statistics):
        return []

    # Filter using stripe-level statistics
    selected_stripes = []
    num_rows_scanned = 0
    for i, stripe_statistics in enumerate(stripes_statistics):
        num_rows_before_stripe = num_rows_scanned
        num_rows_scanned += next(iter(stripe_statistics.values()))[
            "number_of_values"
        ]
        if stripes is not None and i not in stripes:
            continue
        if skip_rows is not None and num_rows_scanned <= skip_rows:
            continue
        else:
            skip_rows = 0
        if (
            skip_rows is not None
            and num_rows is not None
            and num_rows_before_stripe >= skip_rows + num_rows
        ):
            continue
        if filterutils._apply_filters(filters, stripe_statistics):
            selected_stripes.append(i)

    return selected_stripes


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    filters=None,
    stripes=None,
    skip_rows=None,
    num_rows=None,
    use_index=True,
    decimals_as_float=True,
    force_decimal_scale=None,
    timestamp_type=None,
    **kwargs,
):
    """{docstring}"""

    from cudf import DataFrame

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        path_or_data=filepath_or_buffer, compression=None, **kwargs
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if filters is not None:
        filters = filterutils._prepare_filters(filters)
        selected_stripes = _filter_stripes(
            filters,
            filepath_or_buffer,
            stripes,
            skip_rows,
            num_rows,
            already_prepared=True,
        )

        # Return empty if everything was filtered
        if len(selected_stripes) == 0:
            return _make_empty_df(filepath_or_buffer, columns)
        else:
            stripes = selected_stripes

    if filters is not None and (skip_rows is None or num_rows is None):
        # Read file, stripe statistics
        file_statistics, stripes_statistics = read_orc_statistics(
            filepath_or_buffer
        )

        # Get start, end indices of each stripe
        start_index = 0
        end_index = 0
        start_indices = []
        end_indices = []
        stripes_number_of_values = {}
        for i in sorted(stripes):
            number_of_values = next(iter(stripes_statistics[i].values()))[
                "number_of_values"
            ]
            stripes_number_of_values[i] = number_of_values
            end_index += number_of_values - 1
            start_indices.append(start_index)
            end_indices.append(end_index)
            start_index += number_of_values

        # Filter on filter columns
        query_string, local_dict = filterutils._filters_to_query(filters)
        columns_in_predicate = [
            col for conjunction in filters for (col, op, val) in conjunction
        ]
        df = read_orc(
            filepath_or_buffer,
            engine=engine,
            columns=columns_in_predicate,
            filters=None,
            joins=None,
            stripes=stripes,
            skip_rows=skip_rows,
            num_rows=num_rows,
            use_index=use_index,
            decimals_as_float=decimals_as_float,
            force_decimal_scale=force_decimal_scale,
            timestamp_type=timestamp_type,
            **kwargs,
        )
        original_num_rows = len(df)
        df = df.query(query_string, local_dict=local_dict)
        index = df.index
        index_min = index[0].item()
        index_max = index[-1].item()

        # Determine filtered row range
        new_skip_rows = max(
            [
                start_index if start_index <= index_min else 0
                for start_index in start_indices
            ]
        )
        new_num_rows = index_max - new_skip_rows + 1
        new_row_range_num_rows = new_num_rows

        # Determine filtered stripes
        new_stripes = []
        new_stripes_num_rows = 0
        if stripes is None:
            stripes = range(len(stripes_statistics))
        new_stripes_indices = filterutils._apply_filtered_index(
            index, start_indices, end_indices
        )
        new_stripes = [stripes[i] for i in new_stripes_indices]
        for i in new_stripes_indices:
            new_stripes_num_rows += end_indices[i] - start_indices[i] + 1

        # Select either row range or stripes based on # of rows of each
        # Then, update index
        min_num_rows = min(
            original_num_rows, new_row_range_num_rows, new_stripes_num_rows
        )
        if new_row_range_num_rows == min_num_rows:
            skip_rows = new_skip_rows
            num_rows = new_num_rows
            stripes = None
            index -= skip_rows
        elif new_stripes_num_rows == min_num_rows:
            skip_rows = None
            num_rows = None

            # Compute index offsets for each stripe in filtered selection
            stripes_removed = [i for i in stripes if i not in new_stripes]
            stripe_offsets = [0 for _ in new_stripes]
            for i, stripe in enumerate(new_stripes):
                for stripe_removed in stripes_removed:
                    if stripe_removed < stripe:
                        stripe_offsets[i] -= stripes_number_of_values[
                            stripe_removed
                        ]

            # Gather start and end indices for filtered selection of stripes
            new_start_indices = [start_indices[i] for i in new_stripes_indices]
            new_end_indices = [end_indices[i] for i in new_stripes_indices]

            # Apply offsets and update stripes
            index = filterutils._launch_offset_index_ranges(
                index, stripe_offsets, new_start_indices, new_end_indices
            )
            stripes = new_stripes

        # Prepare columns
        if columns is None:
            # TODO: Do something more robust to check for empty type
            # once reading ORC schema is implemented
            columns = [
                [
                    col
                    for col in file_statistics.keys()
                    if "sum" in file_statistics[col]
                    or "count" in file_statistics[col]
                    or "minimum" in file_statistics[col]
                    or "minimumUtc" in file_statistics[col]
                ]
            ]
        elif not isinstance(columns[0], list):
            columns = [columns]

        # Read in payload columns iteratively
        for column_batch in columns:
            column_batch = [
                col for col in column_batch if col not in columns_in_predicate
            ]
            if len(column_batch) > 0:
                df_to_concat = read_orc(
                    filepath_or_buffer,
                    engine=engine,
                    columns=column_batch,
                    filters=None,
                    joins=None,
                    stripes=stripes,
                    skip_rows=skip_rows,
                    num_rows=num_rows,
                    use_index=use_index,
                    decimals_as_float=decimals_as_float,
                    force_decimal_scale=force_decimal_scale,
                    timestamp_type=timestamp_type,
                    **kwargs,
                )
                df_to_concat = df_to_concat.take(index)
                df = cudf.concat([df, df_to_concat], axis=1)

        df.reset_index(inplace=True)
        return df

    if engine == "cudf":
        df = DataFrame._from_table(
            libcudf.orc.read_orc(
                filepath_or_buffer,
                columns,
                stripes,
                skip_rows,
                num_rows,
                use_index,
                decimals_as_float,
                force_decimal_scale,
                timestamp_type,
            )
        )
    else:

        def read_orc_stripe(orc_file, stripe, columns):
            pa_table = orc_file.read_stripe(stripe, columns)
            if isinstance(pa_table, pa.RecordBatch):
                pa_table = pa.Table.from_batches([pa_table])
            return pa_table

        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(filepath_or_buffer)
        if stripes is not None and len(stripes) > 0:
            pa_tables = [
                read_orc_stripe(orc_file, i, columns) for i in stripes
            ]
            pa_table = pa.concat_tables(pa_tables)
        else:
            pa_table = orc_file.read(columns=columns)
        df = cudf.DataFrame.from_arrow(pa_table)

    return df


@ioutils.doc_to_orc()
def to_orc(df, fname, compression=None, enable_statistics=True, **kwargs):
    """{docstring}"""

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=fname, mode="wb", **kwargs
    )
    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            libcudf.orc.write_orc(df, file_obj, compression, enable_statistics)
    else:
        libcudf.orc.write_orc(df, path_or_buf, compression, enable_statistics)
