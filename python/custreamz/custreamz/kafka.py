# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
import custreamz._lib.kafka as libkafka
import confluent_kafka as ck
from custreamz.utils import docutils


class Consumer(object):

    def __init__(self, kafka_configs):
        self.kafka_configs = kafka_configs
        self.kafka_datasource = libkafka.librdkafka(kafka_configs)

    def metadata(self):
        self.kafka_datasource.print_consumer_metadata()

    @docutils.doc_current_configs()
    def current_configs(self):

        """{docstring}"""

        self.kafka_datasource.current_configs()
        
    def read_gdf(
        self,
        lines=True,
        dtype=True,
        compression="infer",
        dayfirst=True,
        byte_range=None,
        topic=None,
        partition=0,
        start=0,
        end=0,
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
            result = self.kafka_datasource.read_gdf(
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
            offset = self.kafka_datasource.get_committed_offset(part.topic, part.partition)
            if offset < 0:
                offset = 0
            toppars.append(ck.TopicPartition(part.topic, part.partition, offset))
        return toppars

    @docutils.doc_get_watermark_offsets()
    def get_watermark_offsets(self, partition, timeout=10000, cached=False):
        """{docstring}"""

        offsets = ()

        try:
            offsets = self.kafka_datasource.get_watermark_offsets(
                topic=partition.topic,
                partition=partition.partition,
                timeout=timeout,
                cached=cached,
            )
        except RuntimeError:
            raise RuntimeError(
                "Unable to connect to Kafka broker"
            )

        if len(offsets) != 2:
            raise RuntimeError(
                "Multiple watermark offsets encountered. "
                + "Only 2 were expected and "
                + str(len(offsets)) + " encountered"
            )

        if offsets[b"low"] < 0:
            offsets[b"low"] = 0

        if offsets[b'high'] < 0:
            offsets[b'high'] = 0

        return offsets[b"low"], offsets[b"high"]

    def commit(self, offsets=None, asynchronous=True):
        for offs in offsets:
            self.kafka_datasource.commit_topic_offset(
                offs.topic, offs.partition, offs.offset, asynchronous
            )

    def produce(self, topic=None, message_val=None, message_key=None):
        if topic == None:
            raise ValueError(
                "You must specify a Topic to produce to"
            )
        
        if message_val == None:
            raise ValueError(
                "The message value is empty. Please specify a message to produce."
            )

        if message_key == None:
            message_key = ""

        return self.kafka_datasource.produce_message(topic, message_val, message_key)

    def flush(self, timeout=10000):
        return self.kafka_datasource.flush(timeout=timeout)

    def unsubscribe(self):
        return self.kafka_datasource.unsubscribe()

    def close(self, timeout=10000):
        return self.kafka_datasource.close(timeout=timeout)

    def test_secure( self,
        lines=True,
        dtype=True,
        compression="infer",
        dayfirst=True,
        byte_range=None,
        topic=None,
        partition=0,
        start=0,
        end=0,
        timeout=10000,
        delimiter="\n",
        *args,
        **kwargs,
    )

        print("Connecting to Kafka")
        result = self.kafka_datasource.read_gdf(
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
        
        print("Result: " + str(result))
