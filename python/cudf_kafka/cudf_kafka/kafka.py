# Copyright (c) 2020, NVIDIA CORPORATION.
from cudf._lib.io.datasource import Datasource
from cudf.utils import ioutils

from cudf_kafka._lib import kafka as libkafka


@ioutils.doc_kafka_datasource()
class KafkaDatasource(Datasource):

    """{docstring}"""

    def __init__(
        self,
        kafka_configs,
        topic,
        partition,
        start_offset,
        end_offset,
        batch_timeout,
        delimiter,
    ):
        self.kafka_configs = kafka_configs
        self.topic = topic
        self.partition = partition
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.batch_timeout = batch_timeout
        self.delimiter = delimiter

        print("kafka.py __init__ called")

        # kafka_confs = {}
        # for key, value in src[0].kafka_configs.items():
        #     kafka_confs[str.encode(key)] = str.encode(value)

        # Create the underlying Cython Datasource object
        # and populate self.c_datasource for source_info
        self.c_datasource = libkafka.create(
            kafka_configs,
            topic.encode(),
            partition,
            start_offset,
            end_offset,
            batch_timeout,
            delimiter.encode(),
        )
