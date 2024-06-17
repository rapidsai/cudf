# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import datetime
import warnings

import pyarrow as pa

import cudf
from cudf._lib import orc as liborc
from cudf.api.types import is_list_like
from cudf.utils import ioutils


def _make_empty_df(filepath_or_buffer, columns):
    from pyarrow import orc

    orc_file = orc.ORCFile(filepath_or_buffer)
    schema = orc_file.schema
    col_names = schema.names if columns is None else columns
    return cudf.DataFrame._from_data(
        data={
            col_name: cudf.core.column.column_empty(
                row_count=0,
                dtype=schema.field(col_name).type.to_pandas_dtype(),
            )
            for col_name in col_names
        }
    )


def _parse_column_statistics(cs, column_statistics_blob):
    # Initialize stats to return and parse stats blob
    column_statistics = {}
    cs.ParseFromString(column_statistics_blob)

    # Load from parsed stats blob into stats to return
    if cs.HasField("numberOfValues"):
        column_statistics["number_of_values"] = cs.numberOfValues
    if cs.HasField("hasNull"):
        column_statistics["has_null"] = cs.hasNull

    if cs.HasField("intStatistics"):
        column_statistics["minimum"] = (
            cs.intStatistics.minimum
            if cs.intStatistics.HasField("minimum")
            else None
        )
        column_statistics["maximum"] = (
            cs.intStatistics.maximum
            if cs.intStatistics.HasField("maximum")
            else None
        )
        column_statistics["sum"] = (
            cs.intStatistics.sum if cs.intStatistics.HasField("sum") else None
        )

    elif cs.HasField("doubleStatistics"):
        column_statistics["minimum"] = (
            cs.doubleStatistics.minimum
            if cs.doubleStatistics.HasField("minimum")
            else None
        )
        column_statistics["maximum"] = (
            cs.doubleStatistics.maximum
            if cs.doubleStatistics.HasField("maximum")
            else None
        )
        column_statistics["sum"] = (
            cs.doubleStatistics.sum
            if cs.doubleStatistics.HasField("sum")
            else None
        )

    elif cs.HasField("stringStatistics"):
        column_statistics["minimum"] = (
            cs.stringStatistics.minimum
            if cs.stringStatistics.HasField("minimum")
            else None
        )
        column_statistics["maximum"] = (
            cs.stringStatistics.maximum
            if cs.stringStatistics.HasField("maximum")
            else None
        )
        column_statistics["sum"] = cs.stringStatistics.sum

    elif cs.HasField("bucketStatistics"):
        column_statistics["true_count"] = cs.bucketStatistics.count[0]
        column_statistics["false_count"] = (
            column_statistics["number_of_values"]
            - column_statistics["true_count"]
        )

    elif cs.HasField("decimalStatistics"):
        column_statistics["minimum"] = (
            cs.decimalStatistics.minimum
            if cs.decimalStatistics.HasField("minimum")
            else None
        )
        column_statistics["maximum"] = (
            cs.decimalStatistics.maximum
            if cs.decimalStatistics.HasField("maximum")
            else None
        )
        column_statistics["sum"] = cs.decimalStatistics.sum

    elif cs.HasField("dateStatistics"):
        column_statistics["minimum"] = (
            datetime.datetime.fromtimestamp(
                datetime.timedelta(cs.dateStatistics.minimum).total_seconds(),
                datetime.timezone.utc,
            )
            if cs.dateStatistics.HasField("minimum")
            else None
        )
        column_statistics["maximum"] = (
            datetime.datetime.fromtimestamp(
                datetime.timedelta(cs.dateStatistics.maximum).total_seconds(),
                datetime.timezone.utc,
            )
            if cs.dateStatistics.HasField("maximum")
            else None
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
    from pyarrow import orc

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc_statistics()
def read_orc_statistics(
    filepaths_or_buffers,
    columns=None,
    **kwargs,
):
    """{docstring}"""

    files_statistics = []
    stripes_statistics = []
    for source in filepaths_or_buffers:
        path_or_buf = ioutils.get_reader_filepath_or_buffer(
            path_or_data=source, **kwargs
        )
        path_or_buf = ioutils._select_single_source(
            path_or_buf, "read_orc_statistics"
        )
        (
            column_names,
            parsed_file_statistics,
            parsed_stripes_statistics,
        ) = liborc.read_parsed_orc_statistics(path_or_buf)

        # Parse file statistics
        file_statistics = {
            column_name: column_stats
            for column_name, column_stats in zip(
                column_names, parsed_file_statistics
            )
            if columns is None or column_name in columns
        }
        files_statistics.append(file_statistics)

        # Parse stripe statistics
        for parsed_stripe_statistics in parsed_stripes_statistics:
            stripe_statistics = {
                column_name: column_stats
                for column_name, column_stats in zip(
                    column_names, parsed_stripe_statistics
                )
                if columns is None or column_name in columns
            }
            if any(
                not parsed_statistics
                for parsed_statistics in stripe_statistics.values()
            ):
                continue
            else:
                stripes_statistics.append(stripe_statistics)

    return files_statistics, stripes_statistics


def _filter_stripes(
    filters, filepath_or_buffer, stripes=None, skip_rows=None, num_rows=None
):
    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # Prepare filters
    filters = ioutils._prepare_filters(filters)

    # Get columns relevant to filtering
    columns_in_predicate = [
        col for conjunction in filters for (col, op, val) in conjunction
    ]

    # Read and parse file-level and stripe-level statistics
    file_statistics, stripes_statistics = read_orc_statistics(
        filepath_or_buffer, columns_in_predicate
    )

    file_stripe_map = []
    for file_stat in file_statistics:
        # Filter using file-level statistics
        if not ioutils._apply_filters(filters, file_stat):
            continue

        # Filter using stripe-level statistics
        selected_stripes = []
        num_rows_scanned = 0
        for i, stripe_statistics in enumerate(stripes_statistics):
            num_rows_before_stripe = num_rows_scanned
            num_rows_scanned += next(
                iter(stripe_statistics.values())
            ).number_of_values
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
            if ioutils._apply_filters(filters, stripe_statistics):
                selected_stripes.append(i)

        file_stripe_map.append(selected_stripes)

    return file_stripe_map


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    filters=None,
    stripes=None,
    skiprows=None,
    num_rows=None,
    use_index=True,
    timestamp_type=None,
    storage_options=None,
    bytes_per_thread=None,
):
    """{docstring}"""
    from cudf import DataFrame

    if skiprows is not None:
        # Do not remove until cuIO team approves its removal.
        warnings.warn(
            "skiprows is deprecated and will be removed.",
            FutureWarning,
        )

    if num_rows is not None:
        # Do not remove until cuIO team approves its removal.
        warnings.warn(
            "num_rows is deprecated and will be removed.",
            FutureWarning,
        )

    # Multiple sources are passed as a list. If a single source is passed,
    # wrap it in a list for unified processing downstream.
    if not is_list_like(filepath_or_buffer):
        filepath_or_buffer = [filepath_or_buffer]

    # Each source must have a correlating stripe list. If a single stripe list
    # is provided rather than a list of list of stripes then extrapolate that
    # stripe list across all input sources
    if stripes is not None:
        if any(not isinstance(stripe, list) for stripe in stripes):
            stripes = [stripes]

        # Must ensure a stripe for each source is specified, unless None
        if not len(stripes) == len(filepath_or_buffer):
            raise ValueError(
                "A list of stripes must be provided for each input source"
            )

    filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
        bytes_per_thread=bytes_per_thread,
        expand_dir_pattern="*.orc",
    )

    if filters is not None:
        selected_stripes = _filter_stripes(
            filters, filepaths_or_buffers, stripes, skiprows, num_rows
        )

        # Return empty if everything was filtered
        if len(selected_stripes) == 0:
            return _make_empty_df(filepaths_or_buffers[0], columns)
        else:
            stripes = selected_stripes

    if engine == "cudf":
        return DataFrame._from_data(
            *liborc.read_orc(
                filepaths_or_buffers,
                columns,
                stripes,
                skiprows,
                num_rows,
                use_index,
                timestamp_type,
            )
        )
    else:
        from pyarrow import orc

        def read_orc_stripe(orc_file, stripe, columns):
            pa_table = orc_file.read_stripe(stripe, columns)
            if isinstance(pa_table, pa.RecordBatch):
                pa_table = pa.Table.from_batches([pa_table])
            return pa_table

        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        if len(filepath_or_buffer) > 1:
            raise NotImplementedError(
                "Using CPU via PyArrow only supports a single a "
                "single input source"
            )

        orc_file = orc.ORCFile(filepath_or_buffer[0])
        if stripes is not None and len(stripes) > 0:
            for stripe_source_file in stripes:
                pa_tables = [
                    read_orc_stripe(orc_file, i, columns)
                    for i in stripe_source_file
                ]
                pa_table = pa.concat_tables(pa_tables)
        else:
            pa_table = orc_file.read(columns=columns)
        df = cudf.DataFrame.from_arrow(pa_table)

    return df


@ioutils.doc_to_orc()
def to_orc(
    df,
    fname,
    compression="snappy",
    statistics="ROWGROUP",
    stripe_size_bytes=None,
    stripe_size_rows=None,
    row_index_stride=None,
    cols_as_map_type=None,
    storage_options=None,
    index=None,
):
    """{docstring}"""

    for col in df._data.columns:
        if isinstance(col, cudf.core.column.CategoricalColumn):
            raise NotImplementedError(
                "Writing to ORC format is not yet supported with "
                "Categorical columns."
            )

    if isinstance(df.index, cudf.CategoricalIndex):
        raise NotImplementedError(
            "Writing to ORC format is not yet supported with "
            "Categorical columns."
        )

    if cols_as_map_type is not None and not isinstance(cols_as_map_type, list):
        raise TypeError("cols_as_map_type must be a list of column names.")

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=fname, mode="wb", storage_options=storage_options
    )
    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            liborc.write_orc(
                df,
                file_obj,
                compression,
                statistics,
                stripe_size_bytes,
                stripe_size_rows,
                row_index_stride,
                cols_as_map_type,
                index,
            )
    else:
        liborc.write_orc(
            df,
            path_or_buf,
            compression,
            statistics,
            stripe_size_bytes,
            stripe_size_rows,
            row_index_stride,
            cols_as_map_type,
            index,
        )


ORCWriter = liborc.ORCWriter
