import pytest
import json
from pprint import pprint

import numpy as np
from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

pa_missing_reason = None
try:
    import pyarrow as pa
    arrow_version = pa.__version__
except ImportError as msg:
    pa_missing_reason = 'Failed to import pyarrow: {}'.format(msg)
    pa = None
    arrow_version = None


def test_arrow_availability():
    # pyarrow is required for generating test data
    assert pa is not None, pa_missing_reason
    assert arrow_version is not None


def get_expected_values():
    np.random.seed(1234)
    names = ['pear', 'orange', 'grape', 'apple']
    means = [0.26, 0.47, 0.36, 0.69]
    dev = 0.25
    for i in range(30):
        j = np.random.choice(range(4))
        yield i, names[j], float(np.random.uniform(low=means[j]-dev,
                                                   high=means[j]+dev))


def make_batch():
    indices, names, weights = zip(*get_expected_values())
    d_index = pa.array(indices).cast(pa.int32())
    unique_names = list(set(names))
    names_map = list(map(unique_names.index, names))
    d_names_map = pa.array(names_map).cast(pa.int32())
    d_names = pa.array(unique_names)
    d_name = pa.DictionaryArray.from_arrays(d_names_map, d_names)
    d_weight = pa.array(weights)
    batch = pa.RecordBatch.from_arrays([d_index, d_name, d_weight],
                                       ['idx', 'name', 'weight'])
    return batch


@pytest.mark.skipif(arrow_version is None,
                    reason='need compatible pyarrow to generate test data')
def test_ipc():

    batch = make_batch()
    schema_bytes = batch.schema.serialize().to_pybytes()
    recordbatches_bytes = batch.serialize().to_pybytes()

    cpu_data = np.ndarray(shape=len(schema_bytes), dtype=np.byte,
                          buffer=bytearray(schema_bytes))

    # Use GDF IPC parser
    schema_ptr = ffi.cast("void*", cpu_data.ctypes.data)
    ipch = libgdf.gdf_ipc_parser_open(schema_ptr, cpu_data.size)

    if libgdf.gdf_ipc_parser_failed(ipch):
        assert 0, str(ffi.string(libgdf.gdf_ipc_parser_get_error(ipch)))
    jsonraw = libgdf.gdf_ipc_parser_get_schema_json(ipch)
    jsontext = ffi.string(jsonraw).decode()
    json_schema = json.loads(jsontext)
    print('json_schema:')
    pprint(json_schema)

    rb_cpu_data = np.ndarray(shape=len(recordbatches_bytes), dtype=np.byte,
                             buffer=bytearray(recordbatches_bytes))
    rb_gpu_data = rmm.to_device(rb_cpu_data)
    del cpu_data

    devptr = ffi.cast("void*", rb_gpu_data.device_ctypes_pointer.value)

    libgdf.gdf_ipc_parser_open_recordbatches(ipch, devptr, rb_gpu_data.size)

    if libgdf.gdf_ipc_parser_failed(ipch):
        assert 0, str(ffi.string(libgdf.gdf_ipc_parser_get_error(ipch)))

    jsonraw = libgdf.gdf_ipc_parser_get_layout_json(ipch)
    jsontext = ffi.string(jsonraw).decode()
    json_rb = json.loads(jsontext)
    print('json_rb:')
    pprint(json_rb)

    offset = libgdf.gdf_ipc_parser_get_data_offset(ipch)

    libgdf.gdf_ipc_parser_close(ipch)

    # Check
    dicts = json_schema['dictionaries']
    assert len(dicts) == 1
    dictdata = dicts[0]['data']['columns'][0]['DATA']
    assert set(dictdata) == {'orange', 'apple', 'pear', 'grape'}

    gpu_data = rb_gpu_data[offset:]

    schema_fields = json_schema['schema']['fields']
    assert len(schema_fields) == 3
    field_names = [f['name'] for f in schema_fields]
    assert field_names == ['idx', 'name', 'weight']

    # check the dictionary id in schema
    assert schema_fields[1]['dictionary']['id'] == dicts[0]['id']

    # Get "idx" column
    idx_buf_off = json_rb[0]['data_buffer']['offset']
    idx_buf_len = json_rb[0]['data_buffer']['length']
    idx_buf = gpu_data[idx_buf_off:][:idx_buf_len]
    assert json_rb[0]['dtype']['name'] == 'INT32'
    idx_size = json_rb[0]['length']
    assert idx_size == 30
    idx_data = np.ndarray(shape=idx_size, dtype=np.int32,
                          buffer=idx_buf.copy_to_host())
    print('idx_data:')
    print(idx_data)

    # Get "name" column
    name_buf_off = json_rb[1]['data_buffer']['offset']
    name_buf_len = json_rb[1]['data_buffer']['length']
    name_buf = gpu_data[name_buf_off:][:name_buf_len]
    assert json_rb[1]['dtype']['name'] == 'DICTIONARY'
    name_size = json_rb[1]['length']
    name_data = np.ndarray(shape=name_size, dtype=np.int32,
                           buffer=name_buf.copy_to_host())
    print('name_data:')
    print(name_data)

    # Get "weight" column
    weight_buf_off = json_rb[2]['data_buffer']['offset']
    weight_buf_len = json_rb[2]['data_buffer']['length']
    weight_buf = gpu_data[weight_buf_off:][:weight_buf_len]
    assert json_rb[2]['dtype']['name'] == 'DOUBLE'
    weight_size = json_rb[2]['length']
    weight_data = np.ndarray(shape=weight_size, dtype=np.float64,
                             buffer=weight_buf.copy_to_host())
    print('weight_data:')
    print(weight_data)

    # verify data
    sortedidx = np.argsort(idx_data)
    idx_data = idx_data[sortedidx]
    name_data = name_data[sortedidx]
    weight_data = weight_data[sortedidx]

    got_iter = zip(idx_data, name_data, weight_data)
    for expected, got in zip(get_expected_values(), got_iter):
        assert expected[0] == got[0]
        assert expected[1] == dictdata[got[1]]
        assert expected[2] == got[2]
