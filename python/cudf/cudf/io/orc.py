# Copyright (c) 2019-2020, NVIDIA CORPORATION.

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
def read_orc_statistics(
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


def _filter_stripes(
    filters,
    filepath_or_buffer,
    stripes=None,
    skip_rows=None,
    num_rows=None,
    statistics=None,
):
    # Prepare filters
    filters = filterutils._prepare_filters(filters)

    # Read and parse file-level and stripe-level statistics
    file_statistics, stripes_statistics = (
        read_orc_statistics(
            filepath_or_buffer, filterutils._columns_in_predicate(filters)
        )
        if statistics is None
        else statistics
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
        # Prepare filters
        filters = filterutils._prepare_filters(filters)

        # Read in statistics
        file_statistics, stripes_statistics = read_orc_statistics(
            filepath_or_buffer, filterutils._columns_in_predicate(filters)
        )

        # Filter stripes using statistics
        selected_stripes = _filter_stripes(
            filters,
            filepath_or_buffer,
            stripes,
            skip_rows,
            num_rows,
            statistics=(file_statistics, stripes_statistics),
        )

        # Return empty if everything was filtered
        if len(selected_stripes) == 0:
            return _make_empty_df(filepath_or_buffer, columns)
        else:
            stripes = selected_stripes

        # TODO: Support case where skip_rows, num_rows specified
        # TODO: Handle case where filtered rows are very far away from each other
        # TODO: Only take this route when columns_in_predicate is a small enough subset of columns
        if skip_rows is None and num_rows is None:
            # Get first and last indices in stripe-level-filtered data
            columns_in_predicate = filterutils._columns_in_predicate(filters)
            q, vals = filterutils._predicate_to_query(filters)
            filtered_row_indices = (
                DataFrame._from_table(
                    libcudf.orc.read_orc(
                        filepath_or_buffer,
                        columns=columns_in_predicate,
                        stripes=stripes,
                        skip_rows=skip_rows,
                        num_rows=num_rows,
                        use_index=use_index,
                        decimals_as_float=decimals_as_float,
                        force_decimal_scale=force_decimal_scale,
                        timestamp_type=timestamp_type,
                    )
                )
                .query(q, local_dict=vals)
                .index
            )
            if filtered_row_indices.empty:
                return _make_empty_df(filepath_or_buffer, columns)
            first_index = int(filtered_row_indices.take(0))
            last_index = int(
                filtered_row_indices.take(filtered_row_indices.size - 1)
            )

            # Align skip_rows with start of stripe first_index is in and set
            # num_rows based on number of rows in all - not just stripe-level-
            # filtered - data
            num_rows_before_stripe = 0
            num_filtered_rows_scanned = 0
            for i, stripe_statistics in enumerate(stripes_statistics):
                num_rows_in_whole_stripe = next(
                    iter(stripe_statistics.values())
                )["number_of_values"]
                if i in stripes:
                    # Find skip_rows
                    num_filtered_rows_before_stripe = num_filtered_rows_scanned
                    num_filtered_rows_scanned += num_rows_in_whole_stripe
                    if skip_rows is None:
                        if (
                            num_filtered_rows_before_stripe - 1
                            < first_index
                            <= num_filtered_rows_scanned - 1
                        ):
                            skip_rows = num_rows_before_stripe

                    # Find num_rows
                    if (
                        num_filtered_rows_before_stripe - 1
                        < last_index
                        <= num_filtered_rows_scanned - 1
                    ):
                        num_rows_in_stripe = (
                            last_index - num_filtered_rows_before_stripe - 1
                        )
                        num_rows = (
                            num_rows_before_stripe
                            + num_rows_in_stripe
                            - skip_rows
                        )
                num_rows_before_stripe += num_rows_in_whole_stripe

            # In this case, we are using num_rows and skip_rows instead of
            # stripes
            stripes = None

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
