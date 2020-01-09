# Copyright (c) 2018, NVIDIA CORPORATION.
import logging

import numpy as np
import pytest

import rmm

import cudf
from cudf.comm.gpuarrow import GpuArrowReader

try:
    import pyarrow as pa

    arrow_version = pa.__version__
except ImportError as msg:
    print("Failed to import pyarrow: {}".format(msg))
    pa = None
    arrow_version = None


def make_gpu_parse_arrow_data_batch():
    np.random.seed(1234)
    lat = np.random.uniform(low=27, high=42, size=23).astype(np.float32)
    lon = np.random.uniform(low=-105, high=-76, size=23).astype(np.float32)

    dest_lat = pa.array(lat)
    dest_lon = pa.array(lon)
    if arrow_version == "0.7.1":
        dest_lat = dest_lat.cast(pa.float32())
        dest_lon = dest_lon.cast(pa.float32())
    batch = pa.RecordBatch.from_arrays(
        [dest_lat, dest_lon], ["dest_lat", "dest_lon"]
    )
    return batch


@pytest.mark.skipif(
    arrow_version is None,
    reason="need compatible pyarrow to generate test data",
)
def test_gpu_parse_arrow_data():
    batch = make_gpu_parse_arrow_data_batch()
    schema_data = batch.schema.serialize()
    recbatch_data = batch.serialize()

    # To ensure compatibility for OmniSci we're going to create this numpy
    # array to be read-only as that's how numpy arrays created from foreign
    # memory buffers will be set
    cpu_schema = np.frombuffer(schema_data, dtype=np.uint8)
    cpu_data = np.frombuffer(recbatch_data, dtype=np.uint8)
    gpu_data = rmm.to_device(cpu_data)
    del cpu_data

    # test reader
    reader = GpuArrowReader(cpu_schema, gpu_data)
    assert reader[0].name == "dest_lat"
    assert reader[1].name == "dest_lon"
    lat = reader[0].data.copy_to_host()
    lon = reader[1].data.copy_to_host()
    assert lat.size == 23
    assert lon.size == 23
    np.testing.assert_array_less(lat, 42)
    np.testing.assert_array_less(27, lat)
    np.testing.assert_array_less(lon, -76)
    np.testing.assert_array_less(-105, lon)

    dct = reader.to_dict()
    np.testing.assert_array_equal(lat, dct["dest_lat"].to_array())
    np.testing.assert_array_equal(lon, dct["dest_lon"].to_array())


expected_values = """
0,orange,0.4713545411053003
1,orange,0.003790919207527499
2,orange,0.4396940888188392
3,apple,0.5693619092183622
4,pear,0.10894215574048405
5,pear,0.09547296520000881
6,orange,0.4123169425191555
7,apple,0.4125838710498503
8,orange,0.1904218750870219
9,apple,0.9289366739893021
10,orange,0.9330387015860205
11,pear,0.46564799732291595
12,apple,0.8573176464520044
13,pear,0.21566885180419648
14,orange,0.9199361970381871
15,orange,0.9819955872277085
16,apple,0.415964752238025
17,grape,0.36941794781567516
18,apple,0.9761832273396152
19,grape,0.16672327312068824
20,orange,0.13311815129622395
21,orange,0.6230693626648358
22,pear,0.7321171864853122
23,grape,0.23106658283660853
24,pear,0.0198404248930919
25,orange,0.4032931749027482
26,grape,0.665861129515741
27,pear,0.10253071509254097
28,orange,0.15243296681892238
29,pear,0.3514868485827787
"""


def get_expected_values():
    lines = filter(lambda x: x.strip(), expected_values.splitlines())
    rows = [ln.split(",") for ln in lines]
    return [(int(idx), name, float(weight)) for idx, name, weight in rows]


def make_gpu_parse_arrow_cats_batch():
    indices, names, weights = zip(*get_expected_values())
    d_index = pa.array(indices).cast(pa.int32())
    unique_names = list(set(names))
    names_map = list(map(unique_names.index, names))
    d_names_map = pa.array(names_map).cast(pa.int32())
    d_names = pa.array(unique_names)
    d_name = pa.DictionaryArray.from_arrays(d_names_map, d_names)
    d_weight = pa.array(weights)
    batch = pa.RecordBatch.from_arrays(
        [d_index, d_name, d_weight], ["idx", "name", "weight"]
    )
    return batch


@pytest.mark.skipif(
    arrow_version is None,
    reason="need compatible pyarrow to generate test data",
)
def test_gpu_parse_arrow_cats():
    batch = make_gpu_parse_arrow_cats_batch()

    stream = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(stream, batch.schema)
    writer.write_batch(batch)
    writer.close()

    schema_bytes = batch.schema.serialize().to_pybytes()
    recordbatches_bytes = stream.getvalue().to_pybytes()[len(schema_bytes) :]

    schema = np.ndarray(
        shape=len(schema_bytes), dtype=np.byte, buffer=bytearray(schema_bytes)
    )
    rb_cpu_data = np.ndarray(
        shape=len(recordbatches_bytes),
        dtype=np.byte,
        buffer=bytearray(recordbatches_bytes),
    )
    rb_gpu_data = rmm.to_device(rb_cpu_data)

    gar = GpuArrowReader(schema, rb_gpu_data)
    columns = gar.to_dict()

    sr_idx = columns["idx"]
    sr_name = columns["name"]
    sr_weight = columns["weight"]

    assert sr_idx.dtype == np.int32
    assert sr_name.dtype == "category"
    assert sr_weight.dtype == np.double
    assert set(sr_name) == {"apple", "pear", "orange", "grape"}

    expected = get_expected_values()
    for i in range(len(sr_idx)):
        got_idx = sr_idx[i]
        got_name = sr_name[i]
        got_weight = sr_weight[i]

        # the serialized data is not of order
        exp_idx, exp_name, exp_weight = expected[got_idx]

        assert got_idx == exp_idx
        assert got_name == exp_name
        np.testing.assert_almost_equal(got_weight, exp_weight)


@pytest.mark.skipif(
    arrow_version is None,
    reason="need compatible pyarrow to generate test data",
)
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_gpu_parse_arrow_int(dtype):

    depdelay = np.array([0, 0, -3, -2, 11, 6, -7, -4, 4, -3], dtype=dtype)
    arrdelay = np.array([5, -3, 1, -2, 22, 11, -12, -5, 4, -9], dtype=dtype)
    d_depdelay = pa.array(depdelay)
    d_arrdelay = pa.array(arrdelay)
    batch = pa.RecordBatch.from_arrays(
        [d_depdelay, d_arrdelay], ["depdelay", "arrdelay"]
    )

    schema_bytes = batch.schema.serialize().to_pybytes()
    recordbatches_bytes = batch.serialize().to_pybytes()

    schema = np.ndarray(
        shape=len(schema_bytes), dtype=np.byte, buffer=bytearray(schema_bytes)
    )

    rb_cpu_data = np.ndarray(
        shape=len(recordbatches_bytes),
        dtype=np.byte,
        buffer=bytearray(recordbatches_bytes),
    )

    rb_gpu_data = rmm.to_device(rb_cpu_data)
    gar = GpuArrowReader(schema, rb_gpu_data)
    columns = gar.to_dict()
    assert columns["depdelay"].dtype == dtype
    assert set(columns) == {"depdelay", "arrdelay"}
    assert list(columns["depdelay"]) == [0, 0, -3, -2, 11, 6, -7, -4, 4, -3]


@pytest.mark.skipif(
    arrow_version is None,
    reason="need compatible pyarrow to generate test data",
)
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_gpu_parse_arrow_timestamps(dtype):
    timestamp = (
        cudf.datasets.timeseries(
            start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={}
        )
        .reset_index()["timestamp"]
        .reset_index(drop=True)
    )
    gdf = cudf.DataFrame({"timestamp": timestamp.astype(dtype)})
    pdf = gdf.to_arrow(preserve_index=False)
    schema_data = pdf.schema.serialize()
    recbatch_data = pdf.to_batches()[0].serialize()

    # To ensure compatibility for OmniSci we're going to create this numpy
    # array to be read-only as that's how numpy arrays created from foreign
    # memory buffers will be set
    cpu_schema = np.frombuffer(schema_data, dtype=np.uint8)
    cpu_data = np.frombuffer(recbatch_data, dtype=np.uint8)
    gpu_data = rmm.to_device(cpu_data)
    del cpu_data

    # test reader
    reader = GpuArrowReader(cpu_schema, gpu_data)
    assert reader[0].name == "timestamp"
    timestamp_arr = reader[0].data.copy_to_host()
    np.testing.assert_array_equal(timestamp_arr, gdf["timestamp"].to_array())
    dct = reader.to_dict()
    np.testing.assert_array_equal(timestamp_arr, dct["timestamp"].to_array())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("cudf.gpuarrow").setLevel(logging.DEBUG)

    test_gpu_parse_arrow_data()
