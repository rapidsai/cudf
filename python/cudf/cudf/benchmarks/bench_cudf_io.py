# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf.tests import dataset_generator as dg

import pytest
import cudf
import numpy as np
import pandavro as pav

common_dtypes_meta = [
    [np.dtype("bool"), 0.0, 250],
    [np.dtype("int16"), 0.2, 480],
    [np.dtype("int32"), 0.3, 177],
    [np.dtype("float32"), 0.17, 25],
    [np.dtype("float64"), 0.18, 104],
    [np.dtype("str"), 0.23, 49],
    [np.dtype("datetime64[ms]"), 0, 178],
]


def bench_avro(tmpdir, benchmark):
    fname = tmpdir.mkdir("cudf_benchmark").join("avro_sample")

    # pandavro doesn't support timedelta, so skipping that dtype
    df = dg.rand_dataframe(common_dtypes_meta, 5000, 0).to_pandas()
    pav.to_avro(str(fname), df)

    benchmark(cudf.read_avro, fname)


@pytest.mark.parametrize("compression", ["infer", "gzip", "bz2"])
def bench_json(compression, tmpdir, benchmark):
    fname = tmpdir.mkdir("cudf_benchmark").join("json_sample")

    dtypes_meta = common_dtypes_meta + [[np.dtype("timedelta64[s]"), 0, 187]]
    df = dg.rand_dataframe(dtypes_meta, 5000, 0).to_pandas()
    df.to_json(fname, compression=compression, lines=True, orient="records")

    benchmark(
        cudf.read_json,
        fname,
        compression=compression,
        lines=True,
        orient="records",
    )
