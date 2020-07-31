# Copyright (c) 2020, NVIDIA CORPORATION.
import confluent_kafka as ck
import pytest

from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("commit_offset", [-1, 0, 1, 1000])
@pytest.mark.parametrize("topic", ["cudf-kafka-test-topic"])
def test_kafka_offset(kafka_client, topic, commit_offset):
    ck_top = ck.TopicPartition(topic, 0, commit_offset)
    offsets = [ck_top]
    kafka_client.commit(offsets=offsets)

    # Get the offsets that were just committed to Kafka
    retrieved_offsets = kafka_client.committed(offsets)

    for off in retrieved_offsets:
        assert_eq(off.topic, offsets[0].topic)
        assert_eq(off.partition, offsets[0].partition)
        assert_eq(off.offset, offsets[0].offset)
