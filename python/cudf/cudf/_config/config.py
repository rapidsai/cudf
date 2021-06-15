from contextlib import contextmanager

_OPTIONS = {
    # nulls compare like NaN/NaT in numeric/datetime comparisons
    "_nulls_compare_like_nans": False
}


@contextmanager
def option_context(*args):
    global _OPTIONS
    original_options = _OPTIONS.copy()
    try:
        for opt, val in zip(args[::2], args[1::2]):
            if opt not in _OPTIONS:
                raise ValueError(f"Invalid option {opt}")
            _OPTIONS[opt] = val
        yield
    finally:
        _OPTIONS = original_options


def get_option(key):
    if key not in _OPTIONS:
        raise ValueError(f"Invalid option {key}")
    return _OPTIONS[key]


def set_option(key, value):
    if key not in _OPTIONS:
        raise ValueError(f"Invalid option {key}")
    _OPTIONS[key] = value
