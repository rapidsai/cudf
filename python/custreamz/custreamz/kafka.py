# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import cudf

import custreamz._libxx.kafka as libkafka
from custreamz.utils import docutils


class KafkaHandle(object):
    def __init__(self, kafka_configs, topics=[], partitions=[]):
        self.kafka_configs = kafka_configs
        if len(topics) == 0:
            raise ValueError(
                "ERROR: You must specify the topic(s) this "
                + "consumer will consume from!"
            )
        if len(partitions) == 0:
            raise ValueError(
                "ERROR: You must specify the topic partitions "
                + "that this consumer will consume from!"
            )
        libkafka.create_kafka_handle(kafka_configs, topics, partitions)

    def metadata(self):
        libkafka.print_consumer_metadata()

    @docutils.doc_current_configs()
    def current_configs(self):

        """{docstring}"""

        libkafka.current_configs()

    @docutils.doc_read_gdf()
    def read_gdf(
        self,
        lines=True,
        dtype=True,
        compression="infer",
        dayfirst=True,
        byte_range=None,
        topic=None,
        partition=0,
        start=-1,
        end=-1,
        timeout=10000,
        delimiter="\n",
        *args,
        **kwargs,
    ):

        """{docstring}"""

        if topic is None:
            raise ValueError(
                "ERROR: You MUST specifiy the topic "
                + "that you want to consume from!"
            )
        else:
            result = libkafka.read_gdf(
                lines=lines,
                dtype=dtype,
                compression=compression,
                dayfirst=dayfirst,
                byte_range=byte_range,
                topic=topic,
                partition=partition,
                start=start,
                end=end,
                timeout=timeout,
                delimiter=delimiter,
            )

            if result is not None:
                return cudf.DataFrame._from_table(result)
            else:
                return cudf.DataFrame()

    def committed(self, topic=None, partitions=[]):
        if topic is None:
            raise ValueError(
                "You must specify a topic to retrieve the offsets from"
            )
        if len(partitions) == 0:
            raise ValueError(
                "You must specify a list of partitions to get the offsets from"
            )

        offsets = libkafka.get_committed_offset(topic, partitions)
        for key, value in offsets.items():
            if value < 0:
                offsets[key] = 0
        return offsets

    @docutils.doc_get_watermark_offsets()
    def get_watermark_offsets(
        self, topic=None, partition=0, *args, **kwargs,
    ):
        """{docstring}"""

        offsets = libkafka.get_watermark_offsets(
            topic=topic, partition=partition
        )

        if len(offsets) != 2:
            raise ValueError(
                "Multiple watermark offsets encountered. "
                + "Only 2 were expected and "
                + str(len(offsets) + " encountered")
            )

        if offsets[b"low"] < 0:
            offsets[b"low"] = 0

        return offsets[b"low"], offsets[b"high"]

    def commit(
        self, topic=None, partition=0, offset=0, *args, **kwargs,
    ):
        libkafka.commit_topic_offset(topic, partition, offset)

    def produce(self, topic=None, message_val=None, message_key=None):
        return libkafka.produce_message(topic, message_val, message_key)

    def flush(self, timeout=10000):
        return libkafka.flush(timeout=timeout)

    def unsubscribe(self):
        return libkafka.unsubscribe()

    def close(self):
        return libkafka.close()
