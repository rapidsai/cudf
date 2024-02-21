# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import os
import subprocess
import argparse
import json
import multiprocessing as mp
import shutil


EXPECTED_VERSION = "16.0.6"
VERSION_REGEX = re.compile(r"  LLVM version ([0-9.]+)")
GPU_ARCH_REGEX = re.compile(r"sm_(\d+)")
SPACES = re.compile(r"\s+")
SEPARATOR = "-" * 16


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-tidy on a project")
    argparser.add_argument("-cdb", type=str,
                           # TODO This is a hack, needs to be fixed
                           default="cpp/build/cuda-11.5.0/clang-tidy/release/compile_commands.clangd.json",
                           help="Path to cmake-generated compilation database"
                           " file. It is always found inside the root of the "
                           "cmake build folder. So make sure that `cmake` has "
                           "been run once before running this script!")
    argparser.add_argument("-exe", type=str, default="clang-tidy",
                           help="Path to clang-tidy exe")
    argparser.add_argument("-ignore", type=str, default="[.]cu$|examples/kmeans/",
                           help="Regex used to ignore files from checking")
    argparser.add_argument("-select", type=str, default=None,
                           help="Regex used to select files for checking")
    argparser.add_argument("-j", type=int, default=-1,
                           help="Number of parallel jobs to launch.")
    args = argparser.parse_args()
    if args.j <= 0:
        args.j = mp.cpu_count()
    args.ignore_compiled = re.compile(args.ignore) if args.ignore else None
    args.select_compiled = re.compile(args.select) if args.select else None
    ret = subprocess.check_output("%s --version" % args.exe, shell=True)
    ret = ret.decode("utf-8")
    version = VERSION_REGEX.search(ret)
    if version is None:
        raise Exception("Failed to figure out clang-tidy version!")
    version = version.group(1)
    if version != EXPECTED_VERSION:
        raise Exception("clang-tidy exe must be v%s found '%s'" % \
                        (EXPECTED_VERSION, version))
    if not os.path.exists(args.cdb):
        raise Exception("Compilation database '%s' missing" % args.cdb)
    return args


def get_all_commands(cdb):
    with open(cdb) as fp:
        return json.load(fp)


def get_gpu_archs(command):
    archs = []
    for loc in range(len(command)):
        if command[loc] != "-gencode":
            continue
        arch_flag = command[loc + 1]
        match = GPU_ARCH_REGEX.search(arch_flag)
        if match is not None:
            archs.append("--cuda-gpu-arch=sm_%s" % match.group(1))
    return archs


def get_index(arr, item):
    try:
        return arr.index(item)
    except:
        return -1


def remove_item(arr, item):
    loc = get_index(arr, item)
    if loc >= 0:
        del arr[loc]
    return loc


def remove_item_plus_one(arr, item):
    loc = get_index(arr, item)
    if loc >= 0:
        del arr[loc + 1]
        del arr[loc]
    return loc


def get_clang_includes(exe):
    dir = os.getenv("CONDA_PREFIX")
    if dir is None:
        ret = subprocess.check_output("which %s 2>&1" % exe, shell=True)
        ret = ret.decode("utf-8")
        dir = os.path.dirname(os.path.dirname(ret))
    header = os.path.join(dir, "include", "ClangHeaders")
    return ["-I", header]


def get_tidy_args(cmd, exe):
    command, file = cmd["command"], cmd["file"]
    is_cuda = file.endswith(".cu")
    command = re.split(SPACES, command)
    # compiler is always clang++!
    command[0] = "clang++"
    # remove compilation and output targets from the original command
    remove_item_plus_one(command, "-c")
    remove_item_plus_one(command, "-o")
    if is_cuda:
        # replace nvcc's "-gencode ..." with clang's "--cuda-gpu-arch ..."
        archs = get_gpu_archs(command)
        command.extend(archs)
        while True:
            loc = remove_item_plus_one(command, "-gencode")
            if loc < 0:
                break
        # "-x cuda" is the right usage in clang
        loc = get_index(command, "-x")
        if loc >= 0:
            command[loc + 1] = "cuda"
        remove_item_plus_one(command, "-ccbin")
        remove_item(command, "--expt-extended-lambda")
        remove_item(command, "--diag_suppress=unrecognized_gcc_pragma")
    command.extend(get_clang_includes(exe))
    return command, is_cuda


def run_clang_tidy_command(tidy_cmd):
    cmd = " ".join(tidy_cmd)
    result = subprocess.run(cmd, check=False, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    status = result.returncode == 0
    if status:
        out = ""
    else:
        out = "CMD: " + cmd
    out += result.stdout.decode("utf-8").rstrip()
    return status, out


def run_clang_tidy(cmd, args):
    command, is_cuda = get_tidy_args(cmd, args.exe)
    tidy_cmd = [args.exe,
                "-header-filter='.*cudf/cpp/(src|include|bench|comms).*'",
                cmd["file"], "--", ]
    tidy_cmd.extend(command)
    status = True
    out = ""
    if is_cuda:
        tidy_cmd.append("--cuda-device-only")
        tidy_cmd.append(cmd["file"])
        ret, out1 = run_clang_tidy_command(tidy_cmd)
        out += out1
        out += "%s" % SEPARATOR
        if not ret:
            status = ret
        tidy_cmd[-2] = "--cuda-host-only"
        ret, out1 = run_clang_tidy_command(tidy_cmd)
        if not ret:
            status = ret
        out += out1
    else:
        tidy_cmd.append(cmd["file"])
        ret, out1 = run_clang_tidy_command(tidy_cmd)
        if not ret:
            status = ret
        out += out1
    return status, out, cmd["file"]


# yikes! global var :(
results = []
def collect_result(result):
    global results
    results.append(result)


def print_result(passed, stdout, file):
    status_str = "PASSED" if passed else "FAILED"
    print(f"{SEPARATOR} File:{file} {status_str} {SEPARATOR}")
    if stdout:
        print(stdout)
        print(f"{SEPARATOR} File:{file} ENDS {SEPARATOR}")


def print_results():
    global results
    status = True
    for passed, stdout, file in results:
        print_result(passed, stdout, file)
        if not passed:
            status = False
    return status


def run_tidy_for_all_files(args, all_files):
    pool = None if args.j == 1 else mp.Pool(args.j)
    # actual tidy checker
    for cmd in all_files:
        # skip files that we don't want to look at
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            continue
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            continue
        if pool is not None:
            pool.apply_async(run_clang_tidy, args=(cmd, args),
                             callback=collect_result)
        else:
            passed, stdout, file = run_clang_tidy(cmd, args)
            collect_result((passed, stdout, file))
    if pool is not None:
        pool.close()
        pool.join()
    return print_results()


def main():
    args = parse_args()
    # Attempt to making sure that we run this script from root of repo always
    if not os.path.exists(".git"):
        raise Exception("This needs to always be run from the root of repo")
    # Check whether clang-tidy exists
    # print(args)
    if "exe" not in args and shutil.which("clang-tidy") is not None:
        print("clang-tidy not found. Exiting...")
        return
    all_files = get_all_commands(args.cdb)
    status = run_tidy_for_all_files(args, all_files)
    if not status:
        raise Exception("clang-tidy failed! Refer to the errors above.")


if __name__ == "__main__":
    main()
