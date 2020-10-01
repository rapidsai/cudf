# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._fuzz_testing import fuzzer


class PythonFuzz(object):
    def __init__(self, func, params=None, data_handle=None, **kwargs):
        self.function = func
        self.data_handler_class = data_handle
        self.fuzz_worker = fuzzer.Fuzzer(
            target=self.function,
            data_handler_class=self.data_handler_class,
            dirs=kwargs.get("dir", None),
            crash_reports_dir=kwargs.get("crash_reports_dir", None),
            regression=kwargs.get("regression", False),
            max_rows_size=kwargs.get("max_rows_size", 100_000),
            max_cols_size=kwargs.get("max_cols_size", 1000),
            runs=kwargs.get("runs", -1),
            max_string_length=kwargs.get("max_string_length", None),
            params=params,
        )

    def __call__(self, *args, **kwargs):
        self.fuzz_worker.start()


# wrap PythonFuzz to allow for deferred calling
def pythonfuzz(function=None, data_handle=None, params=None, **kwargs):
    if function:
        return PythonFuzz(function, params, **kwargs)
    else:

        def wrapper(function):
            return PythonFuzz(function, params, data_handle, **kwargs)

        return wrapper


if __name__ == "__main__":
    PythonFuzz(None)
