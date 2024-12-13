#!/usr/bin/env bash

gdsio_bin=/mnt/gds/tools/gdsio
nvme_dir=/home/coder/cudf/run_benchmark
non_nvme_dir=/home/coder/temp

color_reset='\e[m'
color_green='\e[1;32m'

file_size=4G
io_size=128K
test_duration=60 # sec
device_idx=0
num_threads_per_job=4

echo -e "$color_green--> Sequential write. GDS.$color_reset"
# Sequential write. (-I 1)
# Path: GDS: storage <-> GPU (-x 0).
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 0 -I 1 -T $test_duration

echo -e "$color_green--> Sequential read. GDS.$color_reset"
# Sequential read. (-I 0)
# Path: GDS: storage <-> GPU (-x 0).
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 0 -I 0 -T $test_duration

echo -e "$color_green--> Sequential write. Non-GDS.$color_reset"
# Sequential write. (-I 1)
# Path: Storage <-> CPU <-> GPU (-x 2)
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 2 -I 1 -T $test_duration

echo -e "$color_green--> Sequential read. Non-GDS.$color_reset"
# Sequential read. (-I 0)
# Path: Storage <-> CPU <-> GPU (-x 2)
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 2 -I 0 -T $test_duration





file_size=4G
io_size=4K
test_duration=60 # sec
device_idx=0
num_threads_per_job=4

echo -e "$color_green--> Random write. GDS.$color_reset"
# Random write. (-I 3)
# Path: GDS: storage <-> GPU (-x 0).
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 0 -I 3 -T $test_duration

echo -e "$color_green--> Random read. GDS.$color_reset"
# Random read. (-I 2)
# Path: GDS: storage <-> GPU (-x 0).
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 0 -I 2 -T $test_duration

echo -e "$color_green--> Random write. Non-GDS.$color_reset"
# Random write. (-I 3)
# Path: Storage <-> CPU <-> GPU (-x 2)
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 2 -I 3 -T $test_duration

echo -e "$color_green--> Random read. Non-GDS.$color_reset"
# Random read. (-I 2)
# Path: Storage <-> CPU <-> GPU (-x 2)
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 2 -I 2 -T $test_duration
