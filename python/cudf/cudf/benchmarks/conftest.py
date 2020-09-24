def pytest_addoption(parser):
    parser.addoption("--use_file", action="store", default=True)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.use_file
    if "use_file" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("use_file", [option_value])
