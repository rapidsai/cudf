# Copyright (c) 2020, NVIDIA CORPORATION.
import confluent_kafka as ck
from cudf_kafka._lib.kafka import KafkaDatasource

import cudf

# import custreamz._libxx.kafka as libkafka
from custreamz.utils import docutils


# Base class for anything class that needs to interact with Apache Kafka
class CudfKafkaClient(object):
    def __init__(self, kafka_configs, topic, partition, delimiter):
        self.kafka_configs = kafka_configs
        self.topic = topic
        self.partition = partition
        self.delimiter = delimiter
        print("Base __init__ in CudfKafkaClient invoked")
        kafka_confs = {}
        for key, value in kafka_configs.items():
            kafka_confs[str.encode(key)] = str.encode(value)
        self.kafka_datasource = KafkaDatasource(
            kafka_confs,
            self.topic.encode(),
            self.partition,
            0,
            10,
            10000,
            self.delimiter.encode(),
        )

    def metadata(self):
        self.kafka_datasource.print_consumer_metadata()

    @docutils.doc_current_configs()
    def current_configs(self):

        """{docstring}"""

        self.kafka_datasource.current_configs()

    def unsubscribe(self):
        return self.kafka_datasource.unsubscribe()

    def close(self, timeout=10000):
        return self.kafka_datasource.close(timeout=timeout)


# Kafka Consumer implementation
class Consumer(CudfKafkaClient):
    def __init__(self, kafka_configs, topic, partition, delimiter):
        super().__init__(kafka_configs, topic, partition, delimiter)
        print("__init__ in Consumer invoked")

    @docutils.doc_read_gdf()
    def read_gdf(
        self,
        lines=True,
        dtype=True,
        compression="infer",
        dayfirst=True,
        byte_range=None,
        topic=None,
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
            result = cudf.io.read_csv(
                self.kafka_source,
                lines=lines,
                dtype=dtype,
                compression=compression,
                dayfirst=dayfirst,
                byte_range=byte_range,
            )

            if result is not None:
                return cudf.DataFrame._from_table(result)
            else:
                return cudf.DataFrame()

    def committed(self, partitions, timeout=10000):
        toppars = []
        for part in partitions:
            offset = self.kafka_datasource.get_committed_offset(
                part.topic, part.partition
            )
            if offset < 0:
                offset = 0
            toppars.append(
                ck.TopicPartition(part.topic, part.partition, offset)
            )
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
            raise RuntimeError("Unable to connect to Kafka broker")

        if len(offsets) != 2:
            raise RuntimeError(
                "Multiple watermark offsets encountered. "
                + "Only 2 were expected and "
                + str(len(offsets))
                + " encountered"
            )

        if offsets[b"low"] < 0:
            offsets[b"low"] = 0

        if offsets[b"high"] < 0:
            offsets[b"high"] = 0

        return offsets[b"low"], offsets[b"high"]

    def commit(self, offsets=None, asynchronous=True):
        for offs in offsets:
            self.kafka_datasource.commit_topic_offset(
                offs.topic, offs.partition, offs.offset, asynchronous
            )


# Kafka Producer implementation
class Producer(CudfKafkaClient):
    def __init__(self, kafka_configs, topic, partition, delimiter):
        super().__init__(kafka_configs, topic, partition, delimiter)
        print("__init__ in Producer invoked")

    def produce(self, message_val=None, message_key=None):
        if message_val is None:
            raise ValueError("The message value is empty.")

        if message_key is None:
            message_key = ""

        return self.kafka_datasource.produce_message(
            self.topic, message_val, message_key
        )

    def flush(self, timeout=10000):
        return self.kafka_datasource.flush(timeout=timeout)
