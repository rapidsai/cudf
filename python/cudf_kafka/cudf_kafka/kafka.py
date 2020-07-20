# Copyright (c) 2020, NVIDIA CORPORATION.
from cudf_kafka._lib.io import kafka as libkafka

from cudf._lib.io.datasource import Datasource
from cudf.utils import ioutils


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

        # for key, value in src[0].kafka_configs.items():
        #     kafka_confs[str.encode(key)] = str.encode(value)
        # c_datasource = make_unique[kafka_consumer](kafka_configs,
        #                              topic,
        #                              partition,
        #                              start_offset,
        #                              end_offset,
        #                              batch_timeout,
        #                              delimiter)

        libkafka.C_KafkaDatasource().create()
