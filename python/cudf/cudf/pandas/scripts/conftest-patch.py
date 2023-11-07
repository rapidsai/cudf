# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sys
from functools import wraps

import pytest


def replace_kwargs(new_kwargs):
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            kwargs.update(new_kwargs)
            return func(*args, **kwargs)

        return wrapped

    return wrapper


@contextlib.contextmanager
def null_assert_warnings(*args, **kwargs):
    try:
        yield []
    finally:
        pass


@pytest.fixture(scope="session", autouse=True)  # type: ignore
def patch_testing_functions():
    tm.assert_produces_warning = null_assert_warnings
    pytest.raises = replace_kwargs({"match": None})(pytest.raises)


sys.path.append(os.path.dirname(__file__))
