#!/usr/bin/env bash

set -e -o pipefail

# https://github.com/NVIDIA/MagnumIO/tree/main/gds/samples

# export LD_LIBRARY_PATH=/home/coder/.conda/envs/rapids/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_LOGFILE_PATH=biu_cufile_log.txt
export CUFILE_LOGGING_LEVEL=ERROR

sample_dir=/mnt/profile/MagnumIO/gds/samples

color_reset='\e[m'
color_green='\e[1;32m'

test_idx=1
print_test_idx() {
    echo -e "$color_green--> Test $test_idx$color_reset"
    ((test_idx += 1))
}

print_test_idx
filepath=foo
device_id=0
$sample_dir/cufile_sample_001 $filepath $device_id

print_test_idx
$sample_dir/cufile_sample_002 $filepath $device_id

print_test_idx
filepath_1=foo1
filepath_2=foo2
$sample_dir/cufile_sample_003 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_004 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_005 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_006 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_007

print_test_idx
$sample_dir/cufile_sample_008

print_test_idx
$sample_dir/cufile_sample_009 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_010 $filepath $device_id $device_id

print_test_idx
$sample_dir/cufile_sample_011 $filepath $device_id $device_id

print_test_idx
$sample_dir/cufile_sample_012 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_013 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_014 $filepath_1 $filepath_2 $device_id

print_test_idx
# mode: DeviceMemory = 1, ManagedMemory = 2, HostMemory = 3
mode=1
$sample_dir/cufile_sample_015 $filepath_1 $filepath_2 $device_id $mode

print_test_idx
$sample_dir/cufile_sample_016 $filepath

print_test_idx
$sample_dir/cufile_sample_017 $filepath

print_test_idx
$sample_dir/cufile_sample_018 $filepath

print_test_idx
num_batch_entries=4
$sample_dir/cufile_sample_019 $filepath $device_id $num_batch_entries

print_test_idx
$sample_dir/cufile_sample_020 $filepath $device_id $num_batch_entries

print_test_idx
$sample_dir/cufile_sample_021 $filepath $device_id $num_batch_entries

print_test_idx
non_direct_flag=0
$sample_dir/cufile_sample_022 $filepath $device_id $non_direct_flag

print_test_idx
$sample_dir/cufile_sample_023 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_024 $filepath

print_test_idx
$sample_dir/cufile_sample_025 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_026 $filepath_1 $filepath_2

print_test_idx
$sample_dir/cufile_sample_027 $filepath $device_id $num_batch_entries $non_direct_flag

print_test_idx
$sample_dir/cufile_sample_028 $filepath $device_id

print_test_idx
$sample_dir/cufile_sample_029 $filepath $device_id $num_batch_entries $non_direct_flag

print_test_idx
# Buf Register 0 - register all buffers, 1 - unregistered buffers
buf_register_flag=0
$sample_dir/cufile_sample_030 $filepath $device_id $num_batch_entries $buf_register_flag $non_direct_flag

print_test_idx
$sample_dir/cufile_sample_031 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_032 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_033 $filepath_1 $filepath_2 $device_id

print_test_idx
$sample_dir/cufile_sample_034 $filepath_1 $filepath_2 $device_id
