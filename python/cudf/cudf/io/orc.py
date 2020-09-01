# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import warnings

import pyarrow as pa
from pyarrow import orc as orc

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils
from cudf.utils import filterutils

import pyorc


def _read_stripe_stats(r, stripe_idx, cols):
    stripe = r.read_stripe(stripe_idx)
    stats = []
    num_row_groups = 0
    for col in cols:
        col_stats = stripe[col]._stats
        num_row_groups = len(col_stats)
        for i, row_group_stats in enumerate(col_stats):
            if len(stats) <= i:
                stats.append({})
            stats[i][col] = row_group_stats
    return num_row_groups, stats


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
        # Coerce filters into list of lists of tuples
        if isinstance(filters[0][0], str):
            filters = [filters]

        # Create bytestream and reader
        f = (
            filepath_or_buffer
            if ioutils.is_file_like(filepath_or_buffer)
            else open(filepath_or_buffer, "rb")
        )
        r = pyorc.Reader(f)

        # Get relevant columns
        cols = set()
        filters_with_ids = []
        for conjunction in filters:
            conjunction_with_ids = []
            for i, (col, op, val) in enumerate(conjunction):
                col_id = r.schema.find_column_id(col)
                cols.add(col_id)
                conjunction_with_ids.append((col_id, op, val))
            filters_with_ids.append(conjunction_with_ids)
        cols = sorted(cols)

        # Apply filters to file-level statistics
        file_stats = {}
        for col in cols:
            file_stats[col] = r[col].statistics
        if not filterutils._apply_filters(filters_with_ids, file_stats):
            return _make_empty_df(filepath_or_buffer, columns)

        # Prepare stripes or selection
        stripes = sorted(stripes) if stripes else range(r.num_of_stripes)

        # Determine set of stripes
        # Filter stripes using metadata, skiprows, numrows
        # Compute numrows in stripes
        # Filter row groups using stripe footers, skiprows, numrows
        # If row groups are consecutive, determine skiprows and numrows
        # If row groups are not consecutive, maybe determine skiprows and numrows for each row group and then concatenate
        # Compare numrows in stripes to skiprows and numrows

        # Determine set of stripes
        # Filter stripes using metadata loaded by libcudf, skiprows, numrows
        # Read in selected stripes for columns referenced by predicate
        # Use loc to get rows that pass predicate (TODO: Maybe use where + Numba)
        # Compute set of row ranges to read given skiprows, numrows (TODO: Maybe somehow read in stripes instead in some cases)
        # Read in all data for row ranges and concatenate

        # Read in row-group-level statistics for each stripe for
        # relevant columns
        num_row_groups = [None] * r.num_of_stripes
        row_group_stats = [None] * r.num_of_stripes
        for stripe_idx in stripes:
            (
                curr_stripe_num_row_groups,
                curr_stripe_row_group_stats,
            ) = _read_stripe_stats(r, stripe_idx, cols)
            num_row_groups[stripe_idx] = curr_stripe_num_row_groups
            row_group_stats[stripe_idx] = curr_stripe_row_group_stats
        f.close()

        # Track whether or not all row groups are filtered out
        filtered_everything = False

        # Apply filters to row-group-level statistics
        num_rows_scanned = 0
        num_rows_to_skip = 0
        num_rows_to_read = 0
        filtered_stripes = []
        filtered_stripes_num_rows = 0
        # TODO: Stop assuming that all stripes are consecutive
        for stripe_idx in stripes:
            # Get stats and # of row groups of current stripe
            stats = row_group_stats[stripe_idx]
            curr_stripe_num_row_groups = num_row_groups[stripe_idx]

            # Apply filters to each row group of current stripe
            filter_stripe = True
            num_rows_scanned_in_stripe = 0
            for row_group_idx in range(curr_stripe_num_row_groups):
                row_group_size = next(iter(stats[row_group_idx].values()))[
                    "number_of_values"
                ]
                if filterutils._apply_filters(
                    filters_with_ids, stats[row_group_idx]
                ):
                    # If this is the first row group to pass filters,
                    # update skiprows
                    if num_rows_to_read == 0:
                        num_rows_to_skip = num_rows_scanned
                        num_rows_to_read = row_group_size
                else:
                    filter_stripe = False
                if num_rows_to_read != 0:
                    num_rows_to_read += row_group_size
                num_rows_scanned += row_group_size
                num_rows_scanned_in_stripe += row_group_size

            # Track which stripes passed filters and # of rows in
            # filtered stripes
            if filter_stripe:
                filtered_stripes.append(stripe_idx)
                filtered_stripes_num_rows += num_rows_scanned_in_stripe
        if len(filtered_stripes) == 0:
            filtered_everything = True

        # Update num_rows, skip_rows, and stripes accordingly
        if not num_rows or num_rows_to_read < num_rows:
            skip_rows = num_rows_to_skip
            num_rows = num_rows_to_read
        stripes = (
            [
                stripe_idx
                for stripe_idx in stripes
                if stripe_idx in filtered_stripes
            ]
            if filtered_stripes_num_rows < num_rows
            else []
        )

        # Return empty if everything was filtered
        if filtered_everything:
            return _make_empty_df(filepath_or_buffer, columns)

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
