#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -e

scale_factor=$1
pwd=$(pwd)

# Clone the datafusion repository and apply a patch
# for single threaded data generation so that a
# single parquet file is generated for each table
git clone https://github.com/apache/datafusion.git datafusion
cd datafusion/benchmarks/
git checkout 679a85f
git apply ${pwd}/tpch.patch

# Generate the data
# Currently, we support only scale factor 1 and 10
if [ ${scale_factor} -eq 1 ]; then
    ./bench.sh data tpch
elif [ ${scale_factor} -eq 10 ]; then
    ./bench.sh data tpch10
else
    echo "Unsupported scale factor"
    exit 1
fi

# Correct the datatypes of the parquet files
python3 ${pwd}/correct_datatypes.py data/tpch_sf${scale_factor}
