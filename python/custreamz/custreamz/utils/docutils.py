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
