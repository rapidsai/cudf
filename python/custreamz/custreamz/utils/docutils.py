# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf.utils.docutils import docfmt_partial

_docstring_current_configs = """
Retrieve the current configurations for the underlying KafkaHandle instance

Returns
-------
Dictionary of key/value pairs representing the
librdkafka configurations and their values.

Examples
--------
>>> from custream import kafka
>>> kafka_configs = {
    "metadata.broker.list": "localhost:9092",
    "enable.partition.eof": "true",
    "group.id": "groupid",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "false"
}
>>> consumer = kafka.KafkaHandle(kafka_configs,
topics=["kafka-topic"], partitions=[0]))
>>> configs = consumer.current_configs()
"""
doc_current_configs = docfmt_partial(docstring=_docstring_current_configs)

_docstring_get_watermark_offsets = """
Retrieve the low and high watermark offsets from the Kafka consumer

Returns
-------
Tuple with a [low, high] value

Examples
--------
>>> from custream import kafka
>>> kafka_configs = {
    "metadata.broker.list": "localhost:9092",
    "enable.partition.eof": "true",
    "group.id": "groupid",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "false"
}
>>> consumer = kafka.KafkaHandle(kafka_configs,
topics=["kafka-topic"], partitions=[0]))
>>> low, high = consumer.get_watermark_offsets("kafka-topic", 0)
"""
doc_get_watermark_offsets = docfmt_partial(
    docstring=_docstring_get_watermark_offsets
)

_docstring_read_gdf = """
Load an Avro dataset into a DataFrame

Parameters
----------
filepath_or_buffer : str, path object, bytes, or file-like object
    Either a path to a file (a `str`, `pathlib.Path`, or
    `py._path.local.LocalPath`), URL (including http, ftp, and S3 locations),
    Python bytes of raw binary data, or any object with a `read()` method
    (such as builtin `open()` file handler function or `BytesIO`).
engine : {{ 'cudf', 'fastavro' }}, default 'cudf'
    Parser engine to use.
columns : list, default None
    If not None, only these columns will be read.
skip_rows : int, default None
    If not None, the nunber of rows to skip from the start of the file.
num_rows : int, default None
    If not None, the total number of rows to read.

Returns
-------
DataFrame

Examples
--------
>>> import cudf
>>> df = cudf.read_avro(filename)
>>> df
  num1                datetime text
0  123 2018-11-13T12:00:00.000 5451
1  456 2018-11-14T12:35:01.000 5784
2  789 2018-11-15T18:02:59.000 6117

See Also
--------
cudf.io.csv.read_csv
cudf.io.json.read_json
"""
doc_read_gdf = docfmt_partial(docstring=_docstring_read_gdf)
