#!/usr/bin/env python3

# Run this script with sudo in order to collect performance counters

import os
import subprocess


class test_base:
    def __init__(self):
        self.nsys_bin = "/mnt/nsight-systems-cli/bin/nsys"
        self.test_env = dict()

        # Hot cache (not set) or cold cache (set)
        self.test_env["CUDF_BENCHMARK_DROP_CACHE"] = "true"

        self.test_env["TMPDIR"] = "/home/coder/cudf/run_benchmark"
        # self.test_env["CUFILE_LOGGING_LEVEL"] = "WARN"
        # self.test_env["CUFILE_NVTX"] = "true"
        self.test_env["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
        self.test_env["KVIKIO_COMPAT_MODE"] = "ON"
        self.test_env["KVIKIO_NTHREADS"] = "4"

        self.color_green = '\x1b[1;32m'
        self.color_magenta = '\x1b[1;35m'
        self.color_end = '\x1b[0m'

    def get_name(self):
        return ""

    def get_benchmark_command(self):
        return ""

    def profile(self):
        # Let KVIKIO use POSIX IO
        subtest_env = self.test_env.copy()
        current_setup = self.get_name()

        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"

        print("{}--> {}{}".format(self.color_green, current_setup, self.color_end))

        env_string = ""
        count = 0
        for key, value in subtest_env.items():
            prefix = ""
            if count > 0:
                prefix = ","
            env_string += "{}{}={}".format(prefix, key, value)
            count += 1

        full_command = "{} profile ".format(self.nsys_bin) +\
            "-o /mnt/profile/{} ".format(current_setup) +\
            "-t nvtx,cuda,osrt " +\
            "-f true " +\
            "--backtrace=none " +\
            "--gpu-metrics-devices=0 " +\
            "--gpuctxsw=true " +\
            "--cuda-memory-usage=true " +\
            "--env-var {} ".format(env_string) +\
            self.get_benchmark_command()

        print("{}--> {}{}".format(self.color_magenta, full_command, self.color_end))
        subprocess.run([full_command], shell=True, env=subtest_env)


class test_parquet(test_base):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "parquet"

    def get_benchmark_command(self):
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH"
        my_bench_name = "parquet_read_io_compression"
        bench_command = "{} -d 0 -b {} ".format(my_bin, my_bench_name) + \
            "-a compression_type=SNAPPY -a io_type=HOST_BUFFER -a cardinality=0 -a run_length=1 --min-samples 40"
        return bench_command


class test_orc(test_base):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "orc"

    def get_benchmark_command(self):
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/ORC_READER_NVBENCH"
        my_bench_name = "orc_read_io_compression"
        bench_command = "{} -d 0 -b {} ".format(my_bin, my_bench_name) + \
            "-a compression=SNAPPY -a io=HOST_BUFFER -a cardinality=0 -a run_length=1 --min-samples 40"
        return bench_command


class test_csv(test_base):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "csv"

    def get_benchmark_command(self):
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/CSV_READER_NVBENCH"
        my_bench_name = "csv_read_io"
        bench_command = "{} -d 0 -b {} -a io=HOST_BUFFER --min-samples 40".format(
            my_bin, my_bench_name)
        return bench_command


class test_json(test_base):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "json"

    def get_benchmark_command(self):
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/JSON_READER_NVBENCH"
        my_bench_name = "json_read_io"
        bench_command = "{} -d 0 -b {} -a io=HOST_BUFFER --min-samples 40".format(
            my_bin, my_bench_name)
        return bench_command


if __name__ == '__main__':
    test_list = [test_parquet(),
                 test_orc(),
                 test_csv(),
                 test_json()
                 ]

    for my_test in test_list:
        my_test.profile()
