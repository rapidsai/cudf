# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf.tests import dataset_generator as dg

import pytest
import cudf
import numpy as np
import pandavro as pav

common_dtypes_meta = [
    [np.dtype("bool"), 0.0, 250],
    [np.dtype("int8"), 0.1, 481],
    [np.dtype("int16"), 0.15, 171],
    [np.dtype("int32"), 0.2, 480],
    [np.dtype("int64"), 0.3, 177],
    [np.dtype("uint8"), 0.14, 418],
    [np.dtype("uint16"), 0.12, 48],
    [np.dtype("uint32"), 0.17, 177],
    [np.dtype("float32"), 0.17, 25],
    [np.dtype("float64"), 0.18, 104],
    [np.dtype("str"), 0.23, 49],
    [np.dtype("datetime64[s]"), 0, 178],
    [np.dtype("datetime64[ms]"), 0, 179],
    [np.dtype("datetime64[us]"), 0, 180],
    [np.dtype("datetime64[ns]"), 0, 181],
]


def get_random_dataframe(
    dtype_meta, size_in_bytes=524_288_000, num_columns=64, seed=0
):
    dtype, _, _ = dtype_meta
    item_size = dtype.itemsize
    if dtype.kind in ("O", "U"):
        # string sizes are limited to 4 characters
        item_size = 4

    num_rows = size_in_bytes // num_columns // item_size

    dtype_metas = [dtype_meta] * num_rows

    return dg.rand_dataframe(dtype_metas, num_rows, seed).to_pandas()


@pytest.mark.parametrize("dtype_meta", common_dtypes_meta)
def bench_avro(dtype_meta, tmpdir, benchmark):
    fname = tmpdir.mkdir("cudf_benchmark").join("avro_sample")

    # pandavro doesn't support timedelta, so skipping that dtype
    df = get_random_dataframe(dtype_meta)
    pav.to_avro(str(fname), df)

    benchmark(cudf.read_avro, fname)


@pytest.mark.parametrize(
    "dtype_meta",
    common_dtypes_meta
    + [
        [np.dtype("timedelta64[s]"), 0, 187],
        [np.dtype("timedelta64[ms]"), 0, 187],
        [np.dtype("timedelta64[us]"), 0, 187],
        [np.dtype("timedelta64[ns]"), 0, 187],
    ],
)
@pytest.mark.parametrize("compression", ["infer", "gzip", "bz2"])
def bench_json(dtype_meta, compression, tmpdir, benchmark):
    fname = tmpdir.mkdir("cudf_benchmark").join("json_sample")

    df = get_random_dataframe(dtype_meta)
    df.to_json(fname, compression=compression, lines=True, orient="records")

    benchmark(
        cudf.read_json,
        fname,
        engine="cudf",
        compression=compression,
        lines=True,
        orient="records",
    )
