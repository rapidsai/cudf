import argparse

from . import fuzzer


class PythonFuzz(object):
    def __init__(self, func, data_handle=None):
        self.function = func
        self.data_handler_class = data_handle

    def __call__(self, *args, **kwargs):
        parser = argparse.ArgumentParser(
            description="fuzzer for python packages"
        )
        parser.add_argument(
            "dirs",
            type=str,
            nargs="*",
            help="one or more directories/files to use.",
        )
        parser.add_argument(
            "--exact-artifact-path",
            type=str,
            help="set exact artifact path for crashes/ooms",
        )
        parser.add_argument(
            "--regression",
            type=bool,
            default=False,
            help="run the fuzzer through set of files for "
            "regression or reproduction",
        )
        parser.add_argument(
            "--max-input-size",
            type=int,
            default=1000,
            help="Max input size in bytes",
        )
        parser.add_argument(
            "--runs",
            type=int,
            default=-1,
            help="Number of individual test runs, -1 (the default)"
            "to run indefinitely.",
        )

        args = parser.parse_args()
        f = fuzzer.Fuzzer(
            target=self.function,
            data_handler_class=self.data_handler_class,
            dirs=args.dirs,
            exact_artifact_path=args.exact_artifact_path,
            regression=args.regression,
            max_input_size=args.max_input_size,
            runs=args.runs,
        )
        f.start()


# wrap PythonFuzz to allow for deferred calling
def pythonfuzz(function=None, data_handle=None):
    if function:
        return PythonFuzz(function)
    else:

        def wrapper(function):
            return PythonFuzz(function, data_handle)

        return wrapper


if __name__ == "__main__":
    PythonFuzz()
