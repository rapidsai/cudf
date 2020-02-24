# Copyright (c) 2020, NVIDIA CORPORATION.

import custreamz._libxx.kafka as libkafka


class KafkaHandle(object):
    def __init__(self, kafka_configs):
        self.kafka_configs = kafka_configs
        libkafka.create_kafka_handle(kafka_configs)

    def print_consumer_metadata(self):
        libkafka.print_consumer_metadata()

    def dump_configs(self):
        libkafka.dump_configs()

    def read_gdf(
        lines=True, start=-1, end=-1, timeout=10000, *args, **kwargs,
    ):
        libkafka.read_gdf(lines, start, end, timeout)

    def get_committed_offset(self):
        libkafka.get_committed_offset()

    def get_watermark_offsets(
        topic=None, partition=-1, *args, **kwargs,
    ):
        libkafka.get_watermark_offsets(topic, partition)

    def commit_offsets(*args, **kwargs):
        libkafka.commit_offsets()
