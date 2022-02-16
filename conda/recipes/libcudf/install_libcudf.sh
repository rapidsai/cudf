#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.

./build.sh -v libcudf --allgpuarch --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
