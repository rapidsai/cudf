# Copyright (c) 2022, NVIDIA CORPORATION.
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

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess


EXPECTED_VERSIONS = ("11.1.0",)
VERSION_REGEX = re.compile(r"clang version ([0-9.]+)")
CMAKE_COMPILER_REGEX = re.compile(
    r"^\s*CMAKE_CXX_COMPILER:FILEPATH=(.+)\s*$", re.MULTILINE)
CLANG_COMPILER = "clang++"
GPU_ARCH_REGEX = re.compile(r"sm_(\d+)")
SPACES = re.compile(r"\s+")
XCOMPILER_FLAG = re.compile(r"-((Xcompiler)|(-compiler-options))=?")
XPTXAS_FLAG = re.compile(r"-((Xptxas)|(-ptxas-options))=?")
# any options that may have equal signs in nvcc but not in clang
# add those options here if you find any
OPTIONS_NO_EQUAL_SIGN = ['-isystem']
SEPARATOR = "-" * 8
END_SEPARATOR = "*" * 64


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-tidy on a project")
    argparser.add_argument(
        "-cdb", type=str, default="cpp/build/cuda-11.5.0/clang-tidy-ci/release/compile_commands.clangd.json",
        # "-cdb", type=str, default="compile_commands.json",
        help="Path to cmake-generated compilation database")
    argparser.add_argument(
        "-exe", type=str, default="clang-tidy", help="Path to clang-tidy exe")
    argparser.add_argument(
        "-ignore", type=str, default="(_deps|thrust)",
        help="Regex used to ignore files from checking")
    argparser.add_argument(
        "-select", type=str, default='[.]cpp$',
        help="Regex used to select files for checking")
    argparser.add_argument(
        "-j", type=int, default=-1, help="Number of parallel jobs to launch.")
    argparser.add_argument(
        "-root", type=str, default=None,
        help="Repo root path to filter headers correctly, CWD by default.")
    argparser.add_argument(
        # "-thrust_dir", type=str, default=None,
        "-thrust_dir", type=str, default="/home/cph/dev/rapids/cudf/cpp/build/cuda-11.5.0/clang-tidy-ci/release/_deps/thrust-src",
        help="Pass the directory to a THRUST git repo recent enough for clang.")
    argparser.add_argument(
        "-build_dir", type=str, default=None,
        help="Directory from which compile commands should be called. "
        "By default, directory of compile_commands.json file.")
    args = argparser.parse_args()
    if args.j <= 0:
        args.j = mp.cpu_count()
    args.ignore_compiled = re.compile(args.ignore) if args.ignore else None
    args.select_compiled = re.compile(args.select) if args.select else None
    # we don't check clang's version, it should be OK with any clang
    # recent enough to handle CUDA >= 11
    if not os.path.exists(args.cdb):
        raise Exception("Compilation database '%s' missing" % args.cdb)
    # we assume that this script is run from repo root
    if args.root is None:
        args.root = os.getcwd()
    args.root = os.path.realpath(os.path.expanduser(args.root))
    # we need to have a recent enough cub version for clang to compile
    if args.thrust_dir is None:
        args.thrust_dir = os.path.join(
            os.path.dirname(args.cdb), "thrust_1.15", "src", "thrust_1.15")
    if args.build_dir is None:
        args.build_dir = os.path.dirname(args.cdb)
    if not os.path.isdir(args.thrust_dir):
        raise Exception("Cannot find custom thrust dir '%s" % args.thrust_dir)
    return args


def get_gcc_root(args):
    # first try to determine GCC based on CMakeCache
    cmake_cache = os.path.join(args.build_dir, "CMakeCache.txt")
    if os.path.isfile(cmake_cache):
        with open(cmake_cache) as f:
            content = f.read()
        match = CMAKE_COMPILER_REGEX.search(content)
        if match:
            return os.path.dirname(os.path.dirname(match.group(1)))
    # first fall-back to CONDA prefix if we have a build sysroot there
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_sysroot = os.environ.get("CONDA_BUILD_SYSROOT", "")
    if conda_prefix and conda_sysroot:
        return conda_prefix
    # second fall-back to default g++ install
    default_gxx = shutil.which("g++")
    if default_gxx:
        return os.path.dirname(os.path.dirname(default_gxx))
    raise Exception("Cannot find any g++ install on the system.")

def get_all_commands(cdb):
    with open(cdb, "r") as fp:
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


def get_index(arr, item_options):
    return set(i for i, s in enumerate(arr) for item in item_options
               if s == item)


def remove_items(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        del arr[i]


def remove_items_plus_one(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        if i < len(arr) - 1:
            del arr[i + 1]
        del arr[i]
    idx = set(i for i, s in enumerate(arr) for item in item_options
              if s.startswith(item + "="))
    for i in sorted(idx, reverse=True):
        del arr[i]


def add_cuda_path(command, nvcc):
    nvcc_path = shutil.which(nvcc)
    if not nvcc_path:
        raise Exception("Command %s has invalid compiler %s" % (command, nvcc))
    cuda_root = os.path.dirname(os.path.dirname(nvcc_path))
    command.append('--cuda-path=%s' % cuda_root)


def get_tidy_args(cmd, args):
    command, file = cmd["command"], cmd["file"]
    is_cuda = file.endswith(".cu")
    command = re.split(SPACES, command)
    # get original compiler
    cc_orig = command[0]
    # compiler is always clang++!
    command[0] = "clang++"
    # remove compilation and output targets from the original command
    remove_items_plus_one(command, ["--compile", "-c"])
    remove_items_plus_one(command, ["--output-file", "-o"])
    if is_cuda:
        # include our own cub before anything else
        # (left-most should have highest priority)
        command.insert(1, "-I%s" % args.thrust_dir)
        # replace nvcc's "-gencode ..." with clang's "--cuda-gpu-arch ..."
        archs = get_gpu_archs(command)
        command.extend(archs)
        # provide proper cuda path to clang
        add_cuda_path(command, cc_orig)
  	    # remove all kinds of nvcc flags clang doesn't know about
        remove_items_plus_one(command, [
            "--generate-code",
            "-gencode",
            "--x",
            "-x",
            "--compiler-bindir",
            "-ccbin",
            "--diag_suppress",
            "-diag-suppress",
            "--default-stream",
            "-default-stream",
        ])
        remove_items(command, [
            "-extended-lambda",
            "--extended-lambda",
            "-expt-extended-lambda",
            "--expt-extended-lambda",
            "-expt-relaxed-constexpr",
            "--expt-relaxed-constexpr",
            "--device-debug",
            "-G",
            "--generate-line-info",
            "-lineinfo",
        ])
        # "-x cuda" is the right usage in clang
        command.extend(["-x", "cuda"])
        # we remove -Xcompiler flags: here we basically have to hope for the
        # best that clang++ will accept any flags which nvcc passed to gcc
        for i, c in reversed(list(enumerate(command))):
            new_c = XCOMPILER_FLAG.sub('', c)
            if new_c == c:
                continue
            command[i:i + 1] = new_c.split(',')
        # we also change -Xptxas to -Xcuda-ptxas, always adding space here
        for i, c in reversed(list(enumerate(command))):
                if not c.startswith(opt_eq):
                    continue
                x = c.split('=')
                # we only care about the first `=`
                command[i] = x[0]
                command.insert(i + 1, '='.join(x[1:]))
    command.append('-isystem /usr/include/c++/9')
    command.append('-isystem /usr/include/x86_64-linux-gnu/c++/9')
    command.append(' --gcc-toolchain=/home/cph/dev/rapids/compose/etc/conda/cuda_11.5/envs/rapids')
    # somehow this option gets onto the commandline, it is unrecognized by tidy
    remove_items(command, [
        "--forward-unknown-to-host-compiler",
        "-forward-unknown-to-host-compiler"
    ])
    # do not treat warnings as errors here !
    for i, x in reversed(list(enumerate(command))):
        if x.startswith("-Werror"):
            del command[i]
    # try to figure out which GCC CMAKE used, and tell clang all about it
    command.append("--gcc-toolchain=%s" % get_gcc_root(args))
    return command, is_cuda


def check_output_for_errors(output):
    # there shouldn't really be any allowed errors
    warnings = sum(1 for line in output.splitlines() if line.find("warning:") >= 0)
    errors = [line for line in output.splitlines() if line.find("error:") >= 0]
    return warnings, errors


def run_clang_tidy_command(tidy_cmd):
    cmd = " ".join(tidy_cmd)
    result = subprocess.run(cmd, check=False, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result.stdout = result.stdout.decode("utf-8").strip()
    out = "CMD: " + cmd + "\n"
    out += "EXIT-CODE: %d\n" % result.returncode
    n_warnings, errors = check_output_for_errors(result.stdout)
    status = n_warnings == 0 and not errors
    out += result.stdout
    return status, out, errors


class LockContext(object):
    def __init__(self, lock=None) -> None:
        self._lock = lock
    
    def __enter__(self):
        if self._lock:
            self._lock.acquire()
        return self
    
    def __exit__(self, _, __, ___):
        if self._lock:
            self._lock.release()
        return False  # we don't handle exceptions


def print_result(passed, stdout, file, errors):
    if any(errors):
        raise Exception(
            "File %s: got %d errors:\n%s" % (file, len(errors), stdout))
    status_str = "PASSED" if passed else "FAILED"
    print("%s File:%s %s %s" % (SEPARATOR, file, status_str, SEPARATOR))
    if not passed and stdout:
        print(stdout)
        print("%s\n" % END_SEPARATOR)
        return stdout.splitlines()
    return []


def run_clang_tidy(cmd, args):
    command, is_cuda = get_tidy_args(cmd, args)
    header_filter = "-header-filter='.*cudf/cpp/(include|src|test)/.*(?!brotli_dict).*'"
    tidy_cmd = [args.exe, header_filter, cmd["file"], "--"]
    tidy_cmd.extend(command)
    status = True
    out = ""
    if is_cuda:
        tidy_cmd.append("--cuda-device-only")
        tidy_cmd.append(cmd["file"])
        ret, out1, errors1 = run_clang_tidy_command(tidy_cmd)
        out += out1
        out += "\n%s\n" % SEPARATOR
        status = status and ret
        tidy_cmd[-2] = "--cuda-host-only"
        ret, out1, errors2 = run_clang_tidy_command(tidy_cmd)
        status = status and ret
        out += out1
        errors = errors1 + errors2
    else:
        tidy_cmd.append(cmd["file"])
        ret, out1, errors = run_clang_tidy_command(tidy_cmd)
        status = status and ret
        out += out1
    # we immediately print the result since this is more interactive for user
    with lock:
        lines = print_result(status, out, cmd["file"], errors)
        return status, lines


def parse_results(results):
    return all(r[0] for r in results), [s for r in results for s in r[1]]


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
        results.append(run_clang_tidy(cmd, args))
    return parse_results(results)


def copy_lock(init_lock):
    # this is required to pass locks to pool workers
    # see https://stackoverflow.com/questions/25557686/
    # python-sharing-a-lock-between-processes
    global lock
    lock = init_lock

# mostly used for debugging purposes
def run_sequential(args, all_files):
    status = True
    # actual tidy checker
    for cmd in all_files:
        # skip files that we don't want to look at
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            continue
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            continue
        passed, stdout, file = run_clang_tidy(cmd, args)
        print_result(passed, stdout, file)
        if not passed:
            status = False
    return status

def run_parallel(args, all_files):
    init_lock = LockContext(mp.Lock())
    pool = mp.Pool(args.j, initializer=copy_lock, initargs=(init_lock,))
    results = []
    # actual tidy checker
    for cmd in all_files:
        # skip files that we don't want to look at
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            continue
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            continue
        results.append(pool.apply_async(run_clang_tidy, args=(cmd, args)))
    results_final = [r.get() for r in results]
    pool.close()
    pool.join()
    return parse_results(results_final)


def main():
    args = parse_args()
    # Attempt to making sure that we run this script from root of repo always
    if not os.path.exists(".git"):
        raise Exception("This needs to always be run from the root of repo")
    all_files = get_all_commands(args.cdb)
    # ensure that we use only the real paths
    for cmd in all_files:
        cmd["file"] = os.path.realpath(os.path.expanduser(cmd["file"]))
    if args.j == 1:
        status, lines = run_sequential(args, all_files)
    else:
        status, lines = run_parallel(args, all_files)
    if not status:
        # first get a list of all checks that were run
        ret = subprocess.check_output(args.exe + " --list-checks", shell=True)
        ret = ret.decode("utf-8")
        checks = [line.strip() for line in ret.splitlines()
                  if line.startswith(' ' * 4)]
        max_check_len = max(len(c) for c in checks)
        check_counts = dict()
        content = os.linesep.join(lines)
        for check in checks:
            check_counts[check] = content.count(check)
        sorted_counts = sorted(
            check_counts.items(), key=lambda x: x[1], reverse=True)
        print("Failed {} check(s) in total. Counts as per below:".format(
            sum(1 for _, count in sorted_counts if count > 0)))
        for check, count in sorted_counts:
            if count <= 0:
                break
            n_space = max_check_len - len(check) + 4
            print("{}:{}{}".format(check, ' ' * n_space, count))
        raise Exception("clang-tidy failed! Refer to the errors above.")


if __name__ == "__main__":
    main()
