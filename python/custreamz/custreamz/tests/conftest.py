# Copyright (c) 2020, NVIDIA CORPORATION.
import pytest
from confluent_kafka.admin import AdminClient

from custreamz import kafka

# Test for the prescence of a running kafka broker
kafka_available = True
admin_client = AdminClient(
    {"bootstrap.servers": "localhost:9092", "socket.timeout.ms": 1000}
)
topics = admin_client.list_topics().topics

if not topics:
    kafka_available = False


@pytest.fixture(scope="session")
def kafka_client():
    kafka_configs = {
        "metadata.broker.list": "localhost:9092",
        "enable.partition.eof": "true",
        "group.id": "groupid",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": "false",
    }

    if kafka_available is not True:
        pytest.skip(
            "A running Kafka instance must be available to run these tests"
        )

    return kafka.Consumer(kafka_configs)
