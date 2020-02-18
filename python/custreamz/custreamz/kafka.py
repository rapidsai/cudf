# Copyright (c) 2020, NVIDIA CORPORATION.

import custreamz._lib as libcustreamz

# This would be the function to get the actual offsets ... would replace existing stream logic.
def get_kafka_offsets(
    *args, **kwargs,
):
    """{docstring}"""
    # would call cudf externaldatasource performance logic
    pass


def read_gdf(parser, kafka_configs, *args, **kwargs):
    """{docstring}"""
    # Would call the read performance implementation while still using the existing parsers for code reusability.
    pass


def commit_offsets(offsets, *args, **kwargs):
    """{docstring}"""
    # would call cudf externaldatasource performance logic to commit the offsets
    pass
