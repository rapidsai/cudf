#!/usr/bin/env python3

import os
import subprocess


class test_manager:
    def __init__(self):
        self.test_env = os.environ.copy()

        # Hot cache (not set) or cold cache (set)
        self.test_env["CUDF_BENCHMARK_DROP_CACHE"] = "true"

        self.test_env["TMPDIR"] = "/home/coder/cudf/run_benchmark"
        self.test_env["CUFILE_LOGGING_LEVEL"] = "WARN"

        self.color_green = '\x1b[1;32m'
        self.color_end = '\x1b[0m'

    def run_bench_command(self, subtest_env):
        # Parquet
        # my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH"
        # my_option = "parquet_read_io_compression"
        # full_command = "{} -d 0 -b {} -a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1 --min-samples 40".format(
        #     my_bin, my_option)

        # CSV
        my_bin = "/home/coder/cudf/cpp/build/latest/benchmarks/CSV_READER_NVBENCH"
        my_option = "csv_read_io"
        full_command = "{} -d 0 -b {} -a io=FILEPATH --min-samples 40".format(
            my_bin, my_option)

        subprocess.run([full_command], shell=True,
                       env=subtest_env)

    def use_policy_gds(self):
        # Let cuDF use cuFile API that uses GDS
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "GDS"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "false"
        current_setup = "{}_cufile_compat_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                     subtest_env["CUFILE_ALLOW_COMPAT_MODE"])
        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"
        subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
            current_setup)
        print("{}--> {}{}".format(self.color_green,
                                  current_setup, self.color_end))
        self.run_bench_command(subtest_env)

        # Let cuDF use cuFile API that uses POSIX IO
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "GDS"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "true"
        subtest_env["CUFILE_FORCE_COMPAT_MODE"] = "true"
        current_setup = "{}_cufile_compat_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                     subtest_env["CUFILE_ALLOW_COMPAT_MODE"])
        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"
        subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
            current_setup)
        print("{}--> {}{}".format(self.color_green,
                                  current_setup, self.color_end))
        self.run_bench_command(subtest_env)

    def use_policy_always(self):
        # Let cuDF use cuFile API that uses GDS
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "ALWAYS"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "false"
        current_setup = "{}_cufile_compat_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                     subtest_env["CUFILE_ALLOW_COMPAT_MODE"])
        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"
        subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
            current_setup)
        print("{}--> {}{}".format(self.color_green,
                                  current_setup, self.color_end))
        self.run_bench_command(subtest_env)

        # Let cuDF use cuFile API that uses POSIX IO
        subtest_env = self.test_env.copy()
        subtest_env["LIBCUDF_CUFILE_POLICY"] = "ALWAYS"
        subtest_env["CUFILE_ALLOW_COMPAT_MODE"] = "true"
        subtest_env["CUFILE_FORCE_COMPAT_MODE"] = "true"
        current_setup = "{}_cufile_compat_{}".format(subtest_env["LIBCUDF_CUFILE_POLICY"],
                                                     subtest_env["CUFILE_ALLOW_COMPAT_MODE"])
        if "CUDF_BENCHMARK_DROP_CACHE" in subtest_env:
            current_setup += "_cold"
        else:
            current_setup += "_hot"
        subtest_env["CUFILE_LOGFILE_PATH"] = "cufile_log_{}.txt".format(
            current_setup)
        print("{}--> {}{}".format(self.color_green,
                                  current_setup, self.color_end))
        self.run_bench_command(subtest_env)

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
            self.run_bench_command(subtest_env)

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
            self.run_bench_command(subtest_env)

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
            self.run_bench_command(subtest_env)

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
        self.run_bench_command(subtest_env)


if __name__ == '__main__':
    tm = test_manager()
    tm.use_policy_gds()
    tm.use_policy_always()
    tm.use_policy_kvikio()
    tm.use_policy_off()
