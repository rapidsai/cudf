# Copyright (c) 2020, NVIDIA CORPORATION.
import confluent_kafka as ck
from cudf_kafka._lib.kafka import KafkaDatasource

import cudf

from custreamz.utils import docutils


# Base class for anything class that needs to interact with Apache Kafka
class CudfKafkaClient(object):
    def __init__(self, kafka_configs):
        self.kafka_configs = kafka_configs

        self.kafka_confs = {}
        for key, value in self.kafka_configs.items():
            self.kafka_confs[str.encode(key)] = str.encode(value)

        self.kafka_meta_client = KafkaDatasource(self.kafka_confs)

    @docutils.doc_unsubscribe()
    def unsubscribe(self):

        """{docstring}"""

        return self.kafka_meta_client.unsubscribe()

    @docutils.doc_close()
    def close(self, timeout=10000):

        """{docstring}"""

        return self.kafka_meta_client.close(timeout=timeout)


# Apache Kafka Consumer implementation
class Consumer(CudfKafkaClient):
    def __init__(self, kafka_configs):
        super().__init__(kafka_configs)

    @docutils.doc_read_gdf()
    def read_gdf(
        self,
        topic=None,
        partition=0,
        lines=True,
        start=0,
        end=0,
        batch_timeout=10000,
        delimiter="\n",
        message_format="json",
    ):

        """{docstring}"""

        if topic is None:
            raise ValueError(
                "ERROR: You must specifiy the topic "
                + "that you want to consume from"
            )

        kafka_datasource = KafkaDatasource(
            self.kafka_confs,
            topic.encode(),
            partition,
            start,
            end,
            batch_timeout,
            delimiter.encode(),
        )

        if message_format.lower() == "json":
            result = cudf.io.read_json(self.kafka_datasource)
        elif message_format.lower() == "csv":
            result = cudf.io.read_csv(self.kafka_datasource)
        elif message_format.lower() == "orc":
            result = cudf.io.read_orc(self.kafka_datasource)
        elif message_format.lower() == "avro":
            result = cudf.io.read_avro(self.kafka_datasource)
        elif message_format.lower() == "parquet":
            result = cudf.io.read_parquet(self.kafka_datasource)
        else:
            raise ValueError(
                "Unsupported Kafka Message payload type of: "
                + str(message_format)
            )

        # Close up the cudf datasource instance
        kafka_datasource.unsubscribe()
        kafka_datasource.close()

        if result is not None:
            return cudf.DataFrame._from_table(result)
        else:
            # empty Dataframe
            return cudf.DataFrame()

    @docutils.doc_committed()
    def committed(self, partitions, timeout=10000):

        """{docstring}"""

        toppars = []
        for part in partitions:
            offset = self.kafka_meta_client.get_committed_offset(
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
            offsets = self.kafka_meta_client.get_watermark_offsets(
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

    @docutils.doc_commit()
    def commit(self, offsets=None, asynchronous=True):

        """{docstring}"""

        for offs in offsets:
            self.kafka_meta_client.commit_offset(
                offs.topic, offs.partition, offs.offset, asynchronous
            )
