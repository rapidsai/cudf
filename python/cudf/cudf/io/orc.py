# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import warnings
import datetime

import pyarrow as pa
from pyarrow import orc as orc

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils

import cudf.utils.metadata.orc_column_statistics_pb2 as cs_pb2


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
            datetime.timedelta(cs.dateStatistics.minimumUtc).total_seconds(),
            datetime.timezone.utc,
        )
        column_statistics["maximum"] = datetime.datetime.fromtimestamp(
            datetime.timedelta(cs.dateStatistics.maximumUtc).total_seconds(),
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
    filepath_or_buffer, **kwargs,
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

    # Parse statistics
    cs = cs_pb2.ColumnStatistics()
    file_statistics = {}
    stripes_statistics = []
    for i, raw_file_stats in enumerate(raw_file_statistics):
        parsed_statistics = _parse_column_statistics(cs, raw_file_stats)
        if not parsed_statistics:
            return None
        file_statistics[column_names[i].decode("utf-8")] = parsed_statistics
    for raw_stripe_statistics in raw_stripes_statistics:
        stripe_statistics = {}
        for i, raw_file_stats in enumerate(raw_stripe_statistics):
            parsed_statistics = _parse_column_statistics(cs, raw_file_stats)
            if not parsed_statistics:
                return None
            stripe_statistics[
                column_names[i].decode("utf-8")
            ] = parsed_statistics
        stripes_statistics.append(stripe_statistics)

    return file_statistics, stripes_statistics


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    stripes=None,
    skiprows=None,
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

    if engine == "cudf":
        df = DataFrame._from_table(
            libcudf.orc.read_orc(
                filepath_or_buffer,
                columns,
                stripes,
                skiprows,
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
