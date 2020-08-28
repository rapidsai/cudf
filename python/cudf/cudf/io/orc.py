# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import warnings

import pyarrow as pa
from pyarrow import orc as orc

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils

import pyorc


def _filter_row_group(filters, stats, row_group_idx):
    for conjunction in filters:
        res = True
        for col, op, val in conjunction:
            row_group_stats = stats[col]
            if op == "=" or op == "==":
                if (
                    row_group_stats["minimum"][row_group_idx]
                    and val < row_group_stats["minimum"][row_group_idx]
                ):
                    res = False
                if (
                    row_group_stats["maximum"][row_group_idx]
                    and val > row_group_stats["maximum"][row_group_idx]
                ):
                    res = False
        if res:
            return True
    return False


def _read_row_group_statistics(r, stripe_idx, cols):
    stripe = r.read_stripe(stripe_idx)
    stats = {}
    for col in cols:
        col_row_groups_stats = stripe[col]._stats
        stats[col] = {
            "minimum": [col_row_group_stats['minimum'] for col_row_group_stats in col_row_groups_stats],
            "maximum": [col_row_group_stats['maximum'] for col_row_group_stats in col_row_groups_stats],
            "number_of_values": [col_row_group_stats['number_of_values'] for col_row_group_stats in col_row_groups_stats]
        }
    return stats


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


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
        # Track whether or not everything was filtered
        filtered_everything = False

        # Create bytestream and reader
        f = (
            filepath_or_buffer
            if ioutils.is_file_like(filepath_or_buffer)
            else open(filepath_or_buffer, "rb", buffering=0)
        )
        # m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        r = pyorc.Reader(f, column_names=list(set([col for conjunction in filters for col, op, val in conjunction])))

        # Get relevant columns
        cols = set()
        for conjunction in filters:
            for i, (col, op, val) in enumerate(conjunction):
                col = r.schema.find_column_id(col)
                cols.add(col)
                conjunction[i] = (col, op, val)
        cols = sorted(cols)

        # Prepare stripes or selection
        stripes = sorted(stripes) if stripes else range(r.num_of_stripes)

        # Read in row-group statistics for each stripe for relevant columns
        row_group_stats = {}
        for stripe_idx in stripes:
            row_group_stats[stripe_idx] = _read_row_group_statistics(r, stripe_idx, cols)
        del r
        if isinstance(filepath_or_buffer, str):
            f.close()
        
        return cudf.DataFrame([])

        num_rows_scanned = 0
        num_rows_to_read = 0
        has_row_group_passed = False
        filtered_stripes = []
        filtered_stripes_num_rows = 0
        for stripe_idx in stripes:
            stats = row_group_stats[stripe_idx]
            num_row_groups = len(stats[cols[-1]]["number_of_values"])

            # Apply filters to each row group
            filter_stripe = True
            num_rows_scanned_in_stripe = 0
            for row_group_idx in range(num_row_groups):
                row_group_size = next(iter(stats.values()))["number_of_values"][
                    row_group_idx
                ]
                if _filter_row_group(filters, stats, row_group_idx):
                    # If this is the first row group to pass filters,
                    # update skiprows
                    if not has_row_group_passed:
                        skip_rows = (
                            max(skip_rows, num_rows_scanned)
                            if skip_rows
                            else num_rows_scanned
                        )
                        has_row_group_passed = True
                    else:
                        num_rows_to_read = row_group_size
                else:
                    filter_stripe = False
                num_rows_scanned += row_group_size
                num_rows_scanned_in_stripe += row_group_size
            if filter_stripe:
                filtered_stripes.append(stripe_idx)
                filtered_stripes_num_rows += num_rows_scanned_in_stripe
        if len(stripes) == 0:
            filtered_everything = True
        num_rows = (
            min(num_rows, num_rows_to_read) if num_rows else num_rows_to_read
        )
        stripes = (
            [
                stripe_idx
                for stripe_idx in stripes
                if stripe_idx in filtered_stripes
            ]
            if filtered_stripes_num_rows < num_rows
            else []
        )

        # Return if empty
        if filtered_everything:
            return cudf.DataFrame([])

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
