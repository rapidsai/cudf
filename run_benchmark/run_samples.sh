#!/usr/bin/env bash

# https://github.com/NVIDIA/MagnumIO/tree/main/gds/samples

export LD_LIBRARY_PATH=/home/coder/.conda/envs/rapids/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_LOGFILE_PATH

sample_dir=/mnt/profile/samples


filepath=foo
device_id=0
CUFILE_LOGFILE_PATH=cufile_log_1.txt
$sample_dir/cufile_sample_001 $foo $device_id

filepath=foo
device_id=0
CUFILE_LOGFILE_PATH=cufile_log_2.txt
$sample_dir/cufile_sample_002 $foo $device_id

filepath_1=foo1
filepath_2=foo2
device_id=0
CUFILE_LOGFILE_PATH=cufile_log_3.txt
$sample_dir/cufile_sample_003 $filepath_1 $filepath_2 $device_id

CUFILE_LOGFILE_PATH=cufile_log_4.txt
$sample_dir/cufile_sample_004 foo1 foo2 0

CUFILE_LOGFILE_PATH=cufile_log_5.txt
$sample_dir/cufile_sample_005 foo1 foo2 0

CUFILE_LOGFILE_PATH=cufile_log_6.txt
$sample_dir/cufile_sample_006 foo1 foo2 0

CUFILE_LOGFILE_PATH=cufile_log_7.txt
$sample_dir/cufile_sample_007

CUFILE_LOGFILE_PATH=cufile_log_8.txt
$sample_dir/cufile_sample_008

./cufilesample_009 foo1 foo2
cufile_sample_010: Sample multithreaded example with cuFileAPIs. This sample shows how two threads can share the same CUfileHandle_t. Note: The gpu-id1 and gpu-id2 can be the same GPU.

./cufilesample_010 foo 0 0
cufile_sample_011: Sample multithreaded example with cuFileAPIs without using cuFileBufRegister. Note: The gpu-id1 and gpu-id2 can be the same GPU.

./cufilesample_011 foo 0 0
cufile_sample_012: Sample multithreaded example with cuFileAPIs with usage of SetMaxBar1Size and SetMaxCacheSize APIs. This sample uses cuFileBufRegister per thread.

./cufilesample_012 foo1 foo2
cufile_sample_013: Sample multithreaded example with cuFileAPIs with usage of SetMaxBar1Size and SetMaxCacheSize APIs This sample uses cuFileBufRegister alternately per thread.

./cufilesample_013 foo1 foo2
cufile_sample_014: Sample to use a file using cuFileRead buffer offsets

./cufilesample_014 foo1 foo2 0
cufile_sample_015: Sample file data integrity test with cuFileRead and cuFileWrite with Managed Memory

./cufile_sample_015 foo1 foo2 0 1

cufile_sample_016: Sample to test multiple threads reading data at different file offsets and buffer offset of a memory allocated using single allocation but registered with cuFile at different buffer offsets in each thread.

./cufile_sample_016 <file-path>
cufile_sample_017: Sample to test multiple threads reading data at different file offsets and buffer offsets of a memory allocated using single allocation and single buffer registered with cuFile in main thread.

./cufile_sample_017 <file-path>
cufile_sample_018: This sample shows the usage of fcntl locks with GDS for unaligned writes to achieve atomic transactions.

./cufile_sample_018 <file-path>
Note: Following samples need cuFile library version 11.6 and above.

cufile_sample_019: This sample shows the usage of cuFile Batch API for writes.

./cufile_sample_019 <file-path> <gpuid> <num batch entries>
cufile_sample_020: This sample shows the usage of cuFile Batch API for reads.

./cufile_sample_020 <file-path>  <gpuid> <num batch entries>
cufile_sample_021: This sample shows the usage of cuFile Batch API to cancel I/O after submitting a batch read.

./cufile_sample_021 <file-path>  <gpuid> <num batch entries>
cufile_sample_022: This sample shows the usage of cuFile Batch API to perform cuFileBatchIOGetStatus after submitting a batch read. The non O_DIRECT mode works only with libcufile version 12.2 and above. In this sample, nondirectflag is not a mandatory option

./cufile_sample_022 <file-path>  <gpuid> <nondirectflag>
cufile_sample_023: This sample shows the usage of cuFile API with simple cuMemMap allocations.

./cufile_sample_023  <filepathA> <filepathB>
cufile_sample_024: This sample shows the usage of cuFile API with simple cuMemMap allocations and Thrust.

./cufile_sample_024 <file-path>
cufile_sample_025: This sample shows the usage of cuFile API with simple cuMemMap allocations with resize operation.

./cufile_sample_025  <filepathA> <filepathB>
cufile_sample_026: This sample shows the usage of cuFile API with simple cuMemMap allocations with multiple resize operations.

./cufile_sample_026  <filepathA> <filepathB>
cufile_sample_027: This sample shows cuFileBatchIOSubmit Write Test for unaligned I/O with a variation of files opened in O_DIRECT and non O_DIRECT mode. The non O_DIRECT mode works only with libcufile version 12.2 and above.

./cufile_sample_027 <filepath> <gpuid> <num batch entries> <nondirectflag>
Note: Following samples work only with libcufile version 12.2 and above.

cufile_sample_028: This sample shows the simple usage of cuFileWrite API without O_DIRECT MODE. The non O_DIRECT mode works only with libcufile version 12.2 and above.

./cufile_sample_028 <file-path> <gpuid>
cufile_sample_029: This sample shows usage of cuFileBatchIOSubmit API for writes with various combinations of files opened in regular mode, O_DIRECT mode, unaligned I/O, half unregistered buffers and half registered buffers. This sample has files opened with O_DIRECT and non O_DIRECT mode alternatively in the batch.

./cufile_sample_029 <filepath> <gpuid> <num batch entries> <nondirectflag>
cufile_sample_030: This sample shows cuFileBatchIOSubmit Write Test for combination of unaligned I/O, unregistered buffers and registered buffers, This sample has files opened with O_DIRECT and non O_DIRECT mode alternatively in the batch. This sample cycles batch entries with different kinds of memory (cudaMalloc, cudaMallocHost, malloc, mmap) to files in a single batch.

./cufile_sample_030 <filepath> <gpuid> <num batch entries> <Buf Register 0 - register all buffers, 1 - unregistered buffers> <nondirectflag>
cufile_sample_031: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs using default stream.

./cufile_sample_031 <readfilepath> <writefilepath> <gpuid>
cufile_sample_032: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs.

./cufile_sample_032 <readfilepath> <writefilepath> <gpuid>
cufile_sample_033: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs. This shows how the async apis can be used in a batch mode.

./cufile_sample_033 <readfilepath> <writefilepath> <gpuid>
cufile_sample_034: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs with cufile stream registration. This shows how the async apis can be used in a batch mode.

./cufile_sample_034 <readfilepath> <writefilepath> <gpuid>
