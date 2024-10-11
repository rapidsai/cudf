#!/usr/bin/env bash

color_reset='\e[m'
color_green='\e[1;32m'




gdsio_bin=/mnt/gds/tools/gdsio
gds_stats_bin=/mnt/gds/tools/gds_stats
nvme_dir=/home/coder/cudf/run_benchmark

file_size=4G
io_size=128K
test_duration=10 # sec
device_idx=0
num_threads_per_job=4

echo -e "$color_green--> Sequential write. GDS.$color_reset"
# Sequential write. (-I 1)
# Path: GDS: storage <-> GPU (-x 0).
$gdsio_bin -D $nvme_dir -d $device_idx -w $num_threads_per_job -s $file_size -i $io_size -x 0 -I 1 -T $test_duration &

# $! is the process ID of the job most recently placed into the background
$gds_stats_bin -p $! -l 3
