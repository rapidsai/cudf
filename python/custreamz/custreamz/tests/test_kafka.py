# Copyright (c) 2020, NVIDIA CORPORATION.
import confluent_kafka as ck
import pytest

from cudf.tests.utils import assert_eq

from custreamz import kafka


@pytest.fixture(scope="module")
def client():
    kafka_configs = {
        "metadata.broker.list": "localhost:9092",
        "enable.partition.eof": "true",
        "group.id": "groupid",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": "false",
    }
    return kafka.Consumer(kafka_configs)


@pytest.mark.parametrize("commit_offset", [-1, 0, 1, 1000])
@pytest.mark.parametrize("topic", ["cudf-kafka-test-topic"])
def test_kafka_offset(client, topic, commit_offset):
    ck_top = ck.TopicPartition(topic, 0, commit_offset)
    offsets = [ck_top]
    client.commit(offsets=offsets)

    # Get the offsets that were just committed to Kafka
    retrieved_offsets = client.committed(offsets)

    for off in retrieved_offsets:
        assert_eq(off.topic, offsets[0].topic)
        assert_eq(off.partition, offsets[0].partition)
        assert_eq(off.offset, offsets[0].offset)
