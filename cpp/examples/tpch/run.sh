#!/bin/bash
set -e

query_no=$1
dataset_path=$2

if [ -z "$query_no" ]; then
  echo "Usage: $0 <query_no>"
  exit 1
fi

# Set up environment
export KVIKIO_COMPAT_MODE="on"
export LIBCUDF_CUFILE_POLICY="KVIKIO"

./tpch/build/tpch_q${query_no} ${dataset_path}
