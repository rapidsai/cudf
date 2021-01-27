# Copyright (c) 2020-2021, NVIDIA CORPORATION.


def pytest_addoption(parser):
    parser.addoption("--run_bench", action="store", default=False)
    parser.addoption("--bench_pandas", action="store", default=False)
    parser.addoption("--use_buffer", action="store", default=False)
    parser.addoption("--dataset_dir", action="store", default="")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.use_buffer
    if "use_buffer" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("use_buffer", [option_value])

    option_value = metafunc.config.option.bench_pandas
    if "bench_pandas" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("bench_pandas", [option_value])

    option_value = metafunc.config.option.run_bench
    if "run_bench" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("run_bench", [option_value])

    option_value = metafunc.config.option.dataset_dir
    if "dataset_dir" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("dataset_dir", [option_value])


def pytest_configure(config):
    global option
    option = config.option
