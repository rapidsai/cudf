# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.utils.docutils import docfmt_partial

_docstring_current_configs = """
Retrieve the current configurations for the underlying KafkaDatasource instance

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
Read messages from the underlying KafkaDatasource connection and create
a cudf Dataframe


topic=None,
        partition=0,
        lines=True,
        start=0,
        end=0,
        batch_timeout=10000,
        delimiter="\n",
        message_format="json",

Parameters
----------
topic : str, Name of the Kafka topic that the messages should be read from
parition : int, Partition number on the specified topic that
should be read from
lines : {{ True, False }}, default True, Whether messages should be
treated as individual lines
start : int, default 0, The beginning offset that should be used when
reading a range of messages
end : int, default 0, The last offset that will be read when
reading a range of messages
batch_timeout : int, default 10000, Amount of time to wait on the
reading of the messages from Kafka in Milliseconds
delimiter : str, default "\n", If lines=True this is the delimiter that
will be placed between all messages that are read from Kafka
message_format : {{ 'avro', 'csv', 'json', 'orc', 'parquet' }}, default 'json',
Format of the messages that will be read from Kafka.
This dictates which underlying cudf reader will be invoked the
create the Dataframe.

Returns
-------
DataFrame

"""
doc_read_gdf = docfmt_partial(docstring=_docstring_read_gdf)


_docstring_committed = """
Retrieves the last successfully committed Kafka offset of the
underlying KafkaDatasource connection.

Parameters
----------
partitions : list, Topic/Partition instances that specify the TOPPAR
instances the offsets should be retrieved for
timeout : int, default 10000, Max time to wait on the response from
the Kafka broker in milliseconds

Returns
-------
Tuple of ck.TopicPartition objects

"""
doc_committed = docfmt_partial(docstring=_docstring_committed)


_docstring_commit = """
Takes a list of ck.TopicPartition objects and commits their
offset values to the KafkaDatasource connection

Parameters
----------
offsets : list, ck.TopicPartition objects containing the
Topic/Partition/Offset values to be committed to the Kafka broker
asynchronous : {{ True, False }}, default True, True to wait on
Kafka broker response to commit request and False otherwise

"""
doc_commit = docfmt_partial(docstring=_docstring_commit)


_docstring_unsubscribe = """
Stop all active consumption and remove consumer subscriptions
to topic/partition instances

Parameters
----------
None

"""
doc_unsubscribe = docfmt_partial(docstring=_docstring_unsubscribe)

_docstring_close = """
Close the underlying socket connection to Kafka and clean up system resources

Parameters
----------
None

"""
doc_close = docfmt_partial(docstring=_docstring_close)
