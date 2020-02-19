# Copyright (c) 2020, NVIDIA CORPORATION.

import custreamz._libxx.kafka as libkafka


def __init__(self, name):
    self.name = name


def read_gdf(
    engine="cudf",
    data_type="json",
    lines=True,
    kafka_configs=None,
    partition=None,
    start=-1,
    end=-1,
    *args,
    **kwargs,
):
    libkafka.read_json_example()


def get_watermark_offsets(
    datasource_id=None,
    kafka_configs=None,
    topic=None,
    partition=-1,
    *args,
    **kwargs,
):
    libkafka.get_watermark_offsets(
        datasource_id, kafka_configs, topic, partition
    )


def commit_offsets(*args, **kwargs):
    libkafka.commit_offsets()
