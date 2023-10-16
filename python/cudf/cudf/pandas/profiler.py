# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import inspect
import pickle
import sys
import time
from typing import Union

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .fast_slow_proxy import (
    _FinalProxy,
    _FunctionProxy,
    _IntermediateProxy,
    _MethodProxy,
)

# This text is used in contexts where the profiler is injected into the
# original code. The profiler is injected at the top of the cell, so the line
# numbers in the profiler results are offset by 2.
_profile_injection_text = """\
from cudf.pandas import Profiler
with Profiler() as profiler:
{original_lines}

# Patch the results to shift the line numbers back to the original before the
# profiler injection.
new_results = {{}}

for (lineno, currfile, line), v in profiler._results.items():
    new_results[(lineno - 2, currfile, line)] = v

profiler._results = new_results
profiler.print_stats()
{function_profile_printer}
"""

_cpu_issue_text = """\
This workflow could not be completed entirely on the GPU. \
The following functions required CPU fallback:

{cpu_functions}
"""


def lines_with_profiling(lines, print_function_profile=False):
    """Inject profiling code into the given lines of code."""
    cleaned_lines = "\n".join(
        [(" " * 4) + line.replace("\t", " " * 4) for line in lines]
    )
    return _profile_injection_text.format(
        original_lines=cleaned_lines,
        function_profile_printer="profiler.print_per_func_stats()"
        if print_function_profile
        else "",
    )


class Profiler:
    _IGNORE_LIST = ["Profiler()", "settrace(None)"]

    def __init__(self):
        self._results = {}
        self._per_func_results = {}
        self._currkey = None
        self._timer = {}
        self._currfile = None
        self.start_time = None
        self.end_time = None

    def __enter__(self, *args, **kwargs):
        self.start_time = time.perf_counter()
        self._oldtrace = sys.gettrace()
        # Setting the global trace function with sys.settrace does not affect
        # the current call stack, so in addition to this we must also set the
        # current frame's f_trace attribute as done below.
        sys.settrace(self._tracefunc)

        # Following excerpt from:
        # https://docs.python.org/3/library/sys.html#sys.settrace
        # For more fine-grained usage, it is possible
        # to set a trace function by assigning
        # frame.f_trace = tracefunc explicitly, rather than
        # relying on it being set indirectly via the return
        # value from an already installed trace function
        # Hence we need to perform `f_trace = self._tracefunc`
        # we need to `f_back` because current frame will be
        # of this file.
        frame = inspect.currentframe().f_back
        self._currfile = frame.f_code.co_filename
        self._f_back_oldtrace = frame.f_trace
        frame.f_trace = self._tracefunc
        return self

    def __exit__(self, *args, **kwargs):
        sys.settrace(self._oldtrace)
        inspect.currentframe().f_back.f_trace = self._f_back_oldtrace
        self.end_time = time.perf_counter()

    @staticmethod
    def get_namespaced_function_name(
        func_obj: Union[
            _FunctionProxy,
            _MethodProxy,
            type[_FinalProxy],
            type[_IntermediateProxy],
        ]
    ):
        if isinstance(func_obj, _MethodProxy):
            # Extract classname from method object
            type_name = type(func_obj._xdf_wrapped.__self__).__name__
            return ".".join([type_name, func_obj.__name__])
        elif isinstance(func_obj, _FunctionProxy) or issubclass(
            func_obj, (_FinalProxy, _IntermediateProxy)
        ):
            return func_obj.__name__
        else:
            raise NotImplementedError(
                f"Don't know how to get namespaced name for {func_obj}"
            )

    def _tracefunc(self, frame, event, arg):
        if event == "line" and frame.f_code.co_filename == self._currfile:
            key = "".join(inspect.stack()[1].code_context)
            if not any(
                ignore_word in key for ignore_word in Profiler._IGNORE_LIST
            ):
                self._currkey = (frame.f_lineno, self._currfile, key)
                self._results.setdefault(self._currkey, {})
                self._timer[self._currkey] = time.perf_counter()
        elif (
            event == "call"
            and frame.f_code.co_name == "_fast_slow_function_call"
        ):
            if self._currkey is not None:
                self._timer[self._currkey] = time.perf_counter()

            # Store per-function information for free functions and methods
            frame_locals = inspect.getargvalues(frame).locals
            if (
                isinstance(
                    func_obj := frame_locals["args"][0],
                    (_MethodProxy, _FunctionProxy),
                )
                or isinstance(func_obj, type)
                and issubclass(func_obj, (_FinalProxy, _IntermediateProxy))
            ):
                func_name = self.get_namespaced_function_name(func_obj)
                func_values = self._per_func_results.setdefault(
                    func_name, {"current": [], "finished": []}
                )
                func_values["current"].append(time.perf_counter())
        elif (
            event == "return"
            and frame.f_code.co_name == "_fast_slow_function_call"
            # if arg is None the event is caused by an exception which
            # we should not try and handle
            and arg is not None
        ):
            if self._currkey is not None:
                if arg[1]:  # fast
                    run_time = time.perf_counter() - self._timer[self._currkey]
                    self._results[self._currkey][
                        "gpu_time"
                    ] = run_time + self._results[self._currkey].get(
                        "gpu_time", 0
                    )
                else:
                    run_time = time.perf_counter() - self._timer[self._currkey]
                    self._results[self._currkey][
                        "cpu_time"
                    ] = run_time + self._results[self._currkey].get(
                        "cpu_time", 0
                    )

            frame_locals = inspect.getargvalues(frame).locals
            if (
                isinstance(
                    func_obj := frame_locals["args"][0],
                    (_MethodProxy, _FunctionProxy),
                )
                or isinstance(func_obj, type)
                and issubclass(func_obj, (_FinalProxy, _IntermediateProxy))
            ):
                func_name = self.get_namespaced_function_name(func_obj)
                self._per_func_results[func_name]["finished"].append(
                    (
                        arg[1],
                        time.perf_counter()
                        - self._per_func_results[func_name]["current"].pop(),
                    )
                )

        return self._tracefunc

    @property
    def get_stats(self):
        list_data = []
        for key, val in self._results.items():
            cpu_time = val.get("cpu_time", 0)
            gpu_time = val.get("gpu_time", 0)
            line_no, _, line = key
            list_data.append([line_no, line, gpu_time, cpu_time])

        return list_data

    def print_stats(self):
        table = Table()
        table.add_column("Line no.")
        table.add_column("Line")
        table.add_column("GPU TIME(s)")
        table.add_column("CPU TIME(s)")
        for line_no, line, gpu_time, cpu_time in self.get_stats:
            table.add_row(
                str(line_no),
                Syntax(str(line), "python"),
                "" if gpu_time == 0 else "{:.9f}".format(gpu_time),
                "" if cpu_time == 0 else "{:.9f}".format(cpu_time),
            )
        time_elapsed = self.end_time - self.start_time
        table.title = f"""\n\
        Total time elapsed: {time_elapsed:.3f} seconds

        Stats
        """
        console = Console()
        console.print(table)

    def print_per_func_stats(self):
        cpu_funcs = []
        n_gpu_func_calls = 0
        n_cpu_func_calls = 0
        total_gpu_time = 0
        total_cpu_time = 0

        final_data = {}
        for func_name, func_data in self._per_func_results.items():
            final_data[func_name] = {"cpu": [], "gpu": []}

            for is_gpu, runtime in func_data["finished"]:
                key = "gpu" if is_gpu else "cpu"
                final_data[func_name][key].append(runtime)
                if key == "cpu" and func_name not in cpu_funcs:
                    cpu_funcs.append(func_name)

        table = Table()
        for col in (
            "Function",
            "GPU ncalls",
            "GPU cumtime",
            "GPU percall",
            "CPU ncalls",
            "CPU cumtime",
            "CPU percall",
        ):
            table.add_column(col)

        for func_name, func_data in final_data.items():
            gpu_times = func_data["gpu"]
            cpu_times = func_data["cpu"]
            table.add_row(
                func_name,
                f"{len(gpu_times)}",
                f"{sum(gpu_times)}",
                f"{sum(gpu_times) / max(len(gpu_times), 1)}",
                f"{len(cpu_times)}",
                f"{sum(cpu_times)}",
                f"{sum(cpu_times) / max(len(cpu_times), 1)}",
            )
            total_gpu_time += sum(gpu_times)
            total_cpu_time += sum(cpu_times)
            n_gpu_func_calls += len(gpu_times)
            n_cpu_func_calls += len(cpu_times)

        time_elapsed = self.end_time - self.start_time
        table.title = f"""\n\
        Total time elapsed: {time_elapsed:.3f} seconds
        {n_gpu_func_calls} GPU function calls in {total_gpu_time:.3f} seconds
        {n_cpu_func_calls} CPU function calls in {total_cpu_time:.3f} seconds

        Stats
        """
        console = Console()
        console.print(table)

        if cpu_funcs:
            func_list = "\n".join(f"- {func}" for func in cpu_funcs)
            console.print(Markdown(_cpu_issue_text.format(cpu_functions=func_list)))

            call_to_action = (
                "To request GPU support for any of these functions, "
                "[link=https://github.com/rapidsai/cudf/issues/new?"
                "assignees=&labels=%3F+-+Needs+Triage%2C+feature"
                "+request&projects=&template=profiler_cpu_request.md"
                "&title=%5BFEA%5D]"
                "please file a Github issue"
                "[/link].\n"
            )
            console.print(call_to_action)

    def dump_stats(self, file_name):
        pickle_file = open(file_name, "wb")
        pickle.dump(self, pickle_file)
        pickle_file.close()


def load_stats(file_name):
    pickle_file = open(file_name, "rb")
    return pickle.load(pickle_file)
