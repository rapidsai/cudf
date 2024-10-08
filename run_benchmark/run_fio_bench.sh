#!/usr/bin/env bash

fio_bin=fio
nvme_dir=/home/coder/cudf/run_benchmark
non_nvme_dir=/home/coder/temp

color_reset='\e[m'
color_green='\e[1;32m'

file_size=4G
block_size=128K
test_duration=60 # sec
num_threads_per_file=4
io_depth=32

echo -e "$color_green--> Sequential write.$color_reset"
# Sequential write
$fio_bin --directory=$nvme_dir --name=seq_write_nvme \
--direct=1 --ioengine=libaio --group_reporting \
--numjobs=$num_threads_per_file --iodepth=$io_depth \
--bs=$block_size --runtime=$test_duration --time_based \
--rw=write --size=$file_size

echo -e "$color_green--> Sequential read.$color_reset"
# Sequential read
$fio_bin --directory=$nvme_dir --name=seq_read_nvme \
--direct=1 --ioengine=libaio --group_reporting \
--numjobs=$num_threads_per_file --iodepth=$io_depth \
--bs=$block_size --runtime=$test_duration --time_based \
--rw=read --size=$file_size



file_size=4G
block_size=4K
test_duration=60 # sec
num_threads_per_file=4
io_depth=32

echo -e "$color_green--> Random write.$color_reset"
# Random write
$fio_bin --directory=$nvme_dir --name=rand_write_nvme \
--direct=1 --ioengine=libaio --group_reporting \
--numjobs=$num_threads_per_file --iodepth=$io_depth \
--bs=$block_size --runtime=$test_duration --time_based \
--rw=randwrite --size=$file_size

echo -e "$color_green--> Random read.$color_reset"
# Random read
$fio_bin --directory=$nvme_dir --name=rand_read_nvme \
--direct=1 --ioengine=libaio --group_reporting \
--numjobs=$num_threads_per_file --iodepth=$io_depth \
--bs=$block_size --runtime=$test_duration --time_based \
--rw=randread --size=$file_size