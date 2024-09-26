#!/usr/bin/env bash

export LIBCUDF_CUFILE_POLICY="ALWAYS"
# export CUFILE_ALLOW_COMPAT_MODE=false

ctest --test-dir ~/cudf/cpp/build/latest