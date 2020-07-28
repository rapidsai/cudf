# Copyright (c) 2020, NVIDIA CORPORATION.
import confluent_kafka as ck
from cudf_kafka._lib.kafka import KafkaDatasource

import cudf

from custreamz.utils import docutils


# Base class for anything class that needs to interact with Apache Kafka
class CudfKafkaClient(object):
    def __init__(
        self,
        kafka_configs,
        topic=None,
        partition=0,
        start_offset=0,
        end_offset=0,
        delimiter="/n",
        batch_size=10000,
    ):
        self.kafka_configs = kafka_configs
        self.topic = topic
        self.partition = partition
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.delimiter = delimiter
        self.batch_size = batch_size

        kafka_confs = {}
        for key, value in kafka_configs.items():
            kafka_confs[str.encode(key)] = str.encode(value)
        self.kafka_datasource = KafkaDatasource(
            kafka_confs,
            self.topic.encode(),
            self.partition,
            self.start_offset,
            self.end_offset,
            self.batch_size,
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


# Apache Kafka Consumer implementation
class Consumer(CudfKafkaClient):
    def __init__(
        self,
        kafka_configs,
        topic=None,
        partition=0,
        start_offset=0,
        end_offset=0,
        delimiter="/n",
        batch_size=10000,
    ):
        super().__init__(
            kafka_configs,
            topic=topic,
            partition=partition,
            start_offset=start_offset,
            end_offset=end_offset,
            delimiter=delimiter,
            batch_size=batch_size,
        )

    @docutils.doc_read_gdf()
    def read_gdf(
        self, message_format="json",
    ):

        """{docstring}"""

        if self.topic is None:
            raise ValueError(
                "ERROR: You must specifiy the topic "
                + "that you want to consume from"
            )

        if message_format.lower() == "json":
            result = cudf.io.read_json(self.kafka_source)
        elif message_format.lower() == "csv":
            result = cudf.io.read_csv(self.kafka_source)
        elif message_format.lower() == "orc":
            result = cudf.io.read_orc(self.kafka_source)
        elif message_format.lower() == "avro":
            result = cudf.io.read_avro(self.kafka_source)
        elif message_format.lower() == "parquet":
            result = cudf.io.read_parquet(self.kafka_source)
        else:
            raise ValueError(
                "Unsupported Kafka Message payload type of: "
                + str(message_format)
            )

        if result is not None:
            return cudf.DataFrame._from_table(result)
        else:
            # empty Dataframe
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
            self.kafka_datasource.commit_offset(
                offs.topic, offs.partition, offs.offset, asynchronous
            )
