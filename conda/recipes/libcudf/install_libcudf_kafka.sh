#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.

./build.sh -v libcudf_kafka --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
