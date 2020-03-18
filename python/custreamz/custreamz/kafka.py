# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import confluent_kafka as ck

import cudf

import custreamz._libxx.kafka as libkafka
from custreamz.utils import docutils


class Consumer(object):
    def __init__(self, kafka_configs):
        self.kafka_configs = kafka_configs
        libkafka.create_kafka_consumer(kafka_configs)

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

    def committed(self, partitions, timeout=10000):
        toppars = []
        for part in partitions:
            offset = libkafka.get_committed_offset(part.topic, part.partition)
            if offset < 0:
                offset = 0
            toppars.append(ck.TopicPartition(part.topic, part.partition, offset))
        return toppars

    @docutils.doc_get_watermark_offsets()
    def get_watermark_offsets(self, partition, timeout=10000, cached=False):
        """{docstring}"""

        offsets = libkafka.get_watermark_offsets(
            topic=partition.topic,
            partition=partition.partition,
            timeout=timeout,
            cached=cached,
        )

        if len(offsets) != 2:
            raise ValueError(
                "Multiple watermark offsets encountered. "
                + "Only 2 were expected and "
                + str(len(offsets)) + " encountered")
            )

        if offsets[b"low"] < 0:
            offsets[b"low"] = 0

        if offsets[b'high'] < 0:
            offsets[b'high'] = 0

        return offsets[b"low"], offsets[b"high"]

    def commit(self, offsets=None, asynchronous=True):
        for offs in offsets:
            libkafka.commit_topic_offset(
                offs.topic, offs.partition, offs.offset, asynchronous
            )

    def produce(self, topic=None, message_val=None, message_key=None):
        return libkafka.produce_message(topic, message_val, message_key)

    def flush(self, timeout=10000):
        return libkafka.flush(timeout=timeout)

    def unsubscribe(self):
        return libkafka.unsubscribe()

    def close(self):
        return libkafka.close()
