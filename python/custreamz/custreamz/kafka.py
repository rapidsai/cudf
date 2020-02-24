# Copyright (c) 2020, NVIDIA CORPORATION.

import custreamz._libxx.kafka as libkafka


class KafkaHandle(object):
    def __init__(self, kafka_configs):
        self.kafka_configs = kafka_configs
        libkafka.create_kafka_handle(kafka_configs)

    def print_consumer_metadata():
        libkafka.print_consumer_metadata()

    def dump_configs():
        libkafka.dump_configs()

    def read_gdf(
        data_format="blob",
        lines=True,
        partition=None,
        start=-1,
        end=-1,
        *args,
        **kwargs,
    ):
        libkafka.read_gdf()

    def get_watermark_offsets(
        datasource_id=None, topic=None, partition=-1, *args, **kwargs,
    ):
        libkafka.get_watermark_offsets(datasource_id, topic, partition)

    def commit_offsets(*args, **kwargs):
        libkafka.commit_offsets()
