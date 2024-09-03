# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import socket

import pytest

from custreamz import kafka


@pytest.fixture(scope="session")
def kafka_client():
    # Check for the existence of a kafka broker
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect_ex(("localhost", 9092))
        s.shutdown(2)
        s.close()
    except Exception:
        pytest.skip(
            "A running Kafka instance must be available to run these tests"
        )

    kafka_configs = {
        "metadata.broker.list": "localhost:9092",
        "enable.partition.eof": "true",
        "group.id": "groupid",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": "false",
    }

    return kafka.Consumer(kafka_configs)
