# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.utils import ioutils


@ioutils.doc_kafka_datasource()
class KafkaDatasource(object):

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
