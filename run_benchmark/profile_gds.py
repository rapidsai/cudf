#!/usr/bin/env python3

# Run this script with sudo in order to collect performance counters

import os
import subprocess


class test_manager:
    def __init__(self):
        self.test_env = dict()

        # Hot cache (not set) or cold cache (set)
        self.test_env["CUDF_BENCHMARK_DROP_CACHE"] = "true"

        self.test_env["TMPDIR"] = "/home/coder/cudf/run_benchmark"
        self.test_env["CUFILE_LOGGING_LEVEL"] = "WARN"
        self.test_env["CUFILE_NVTX"] = "true"

        self.color_green = '\x1b[1;32m'
        self.color_end = '\x1b[0m'

    def run_bench_command(self, subtest_env, current_setup):
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH"
        my_option = "parquet_read_io_compression"
        nsys_bin = "/mnt/nsight-systems-cli/bin/nsys"

        env_string = ""
        count = 0
        for key, value in subtest_env.items():
            prefix = ""
            if count > 0:
                prefix = ","
            env_string += "{}{}={}".format(prefix, key, value)
            count += 1

        full_command = "{} profile ".format(nsys_bin) +\
            "-o /mnt/profile/{} ".format(current_setup) +\
            "-t nvtx,cuda,osrt " +\
            "-f true " +\
            "--backtrace=none " +\
            "--gpu-metrics-devices=0 " +\
            "--gpuctxsw=true " +\
            "--cuda-memory-usage=true " +\
            "--env-var {} ".format(env_string) +\
            "{} -d 0 -b {} ".format(my_bin, my_option) +\
            "-a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1 --run-once"

        print(full_command)
        subprocess.run([full_command], shell=True,
                       env=subtest_env)

    def use_policy_kvikio(self):
        num_threads_options = [1, 8]

        # Let KVIKIO use cuFile API that uses GDS
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
        subtest_env["KVIKIO_COMPAT_MODE"] = "off"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "false"
        for num_threads in num_threads_options:
            subtest_env["KVIKIO_NTHREADS"] = str(num_threads)
            current_setup = "{}_kvik_compat_{}_cufile_compat_{}_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                                           subtest_env["KVIKIO_COMPAT_MODE"],
                                                                           subtest_env["CUFILE_ALLOW_COMPAT_MODE"],
                                                                           subtest_env["KVIKIO_NTHREADS"])
            if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
                current_setup += "_cold"
            else:
                current_setup += "_hot"
            subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
                current_setup)
            print("{}--> {}{}".format(self.color_green,
                  current_setup, self.color_end))
            self.run_bench_command(subtest_env, current_setup)

        # Let KVIKIO use cuFile API that uses POSIX IO
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
        subtest_env["KVIKIO_COMPAT_MODE"] = "off"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "true"
        subtest_env["CUFILE_FORCE_COMPAT_MODE"] = "true"
        for num_threads in num_threads_options:
            subtest_env["KVIKIO_NTHREADS"] = str(num_threads)
            current_setup = "{}_kvik_compat_{}_cufile_compat_{}_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                                           subtest_env["KVIKIO_COMPAT_MODE"],
                                                                           subtest_env["CUFILE_ALLOW_COMPAT_MODE"],
                                                                           subtest_env["KVIKIO_NTHREADS"])
            if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
                current_setup += "_cold"
            else:
                current_setup += "_hot"
            subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
                current_setup)
            print("{}--> {}{}".format(self.color_green,
                  current_setup, self.color_end))
            self.run_bench_command(subtest_env, current_setup)

        # Let KVIKIO use POSIX IO
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
        subtest_env["KVIKIO_COMPAT_MODE"] = "on"
        for num_threads in num_threads_options:
            subtest_env["KVIKIO_NTHREADS"] = str(num_threads)
            current_setup = "{}_kvik_compat_{}_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                          subtest_env["KVIKIO_COMPAT_MODE"],
                                                          subtest_env["KVIKIO_NTHREADS"])
            if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
                current_setup += "_cold"
            else:
                current_setup += "_hot"
            subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
                current_setup)
            print("{}--> {}{}".format(self.color_green,
                  current_setup, self.color_end))
            self.run_bench_command(subtest_env, current_setup)

    def use_policy_off(self):
        # Let cuDF use POSIX IO
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "OFF"
        current_setup = "{}".format(
            subtest_env["LIBCUDF_CUFILE_POLICY"])
        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"
        subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
            current_setup)
        print("{}--> {}{}".format(self.color_green,
                                  current_setup, self.color_end))
        self.run_bench_command(subtest_env, current_setup)


if __name__ == '__main__':
    tm = test_manager()
    tm.use_policy_kvikio()
    tm.use_policy_off()
