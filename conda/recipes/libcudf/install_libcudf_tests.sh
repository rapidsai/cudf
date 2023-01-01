#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.

cmake --install cpp/build --component testing
cmake --install cpp/libcudf_kafka/build --component testing
