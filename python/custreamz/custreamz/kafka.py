# Copyright (c) 2020, NVIDIA CORPORATION.

import custreamz._libxx as libcustreamz


# This would be the function to get the actual offsets
def get_kafka_offsets(
    *args, **kwargs,
):
    """{docstring}"""
    # would call cudf externaldatasource performance logic
    pass


# def read_json(
#     kafka_configs=kafka_configs,
#     partition=partition,
#     start=low,
#     end=high,
#     *args,
#     **kwargs,
# ):
#     """{docstring}"""
#     pass


def commit_offsets(*args, **kwargs):
    """{docstring}"""

    libcustreamz.commit_offsets()
    pass
