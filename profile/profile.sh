#!/usr/bin/env bash

NSYS_BIN="/usr/local/bin/nsys"
PROGRAM="/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH"
ARGS="-d 0 -b 1 -a io_type=FILEPATH -a compression_type=SNAPPY -a cardinality=0 -a run_length=1"
# ARGS="-d 0 -b 1 -a compression_type=SNAPPY -a cardinality=0 -a run_length=1"

NC='\e[m' # Reset
GREEN='\e[1;32m'

echo -e "${GREEN}--> 8 thread${NC}"
${NSYS_BIN} profile -o rep_8_thread \
--trace=cuda,nvtx,osrt \
--force-overwrite=true \
--backtrace=none \
--gpu-metrics-device=all \
--gpuctxsw=true \
--cuda-memory-usage=true \
--env-var KVIKIO_NTHREADS=8 \
${PROGRAM} ${ARGS}


echo -e "${GREEN}--> 1 thread${NC}"
${NSYS_BIN} profile -o rep_1_thread \
--trace=cuda,nvtx,osrt \
--force-overwrite=true \
--backtrace=none \
--gpu-metrics-device=all \
--gpuctxsw=true \
--cuda-memory-usage=true \
--env-var KVIKIO_NTHREADS=1 \
${PROGRAM} ${ARGS}

