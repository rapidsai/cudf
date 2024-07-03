#!/bin/bash
set -e

query_no=$1
dataset_dir=$2
use_memory_pool=$3

if [ -z "$query_no" ]; then
  echo "Usage: $0 <query_no> <dataset_dir> <use_memory_pool>"
  exit 1
fi

# Set up environment
export KVIKIO_COMPAT_MODE="on"
export LIBCUDF_CUFILE_POLICY="KVIKIO"

./tpch/build/tpch_q${query_no} ${dataset_dir} ${use_memory_pool}
