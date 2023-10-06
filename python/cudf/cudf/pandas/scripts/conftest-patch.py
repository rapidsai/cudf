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

import contextlib
import os
import sys
from functools import wraps


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
    tm.assert_produces_warning = replace_kwargs({"check_stacklevel": False})(
        tm.assert_produces_warning
    )
    tm.assert_series_equal = replace_kwargs({"check_dtype": False})(
        tm.assert_series_equal
    )
    tm.assert_frame_equal = replace_kwargs({"check_dtype": False})(
        tm.assert_frame_equal
    )
    tm.assert_produces_warning = null_assert_warnings


sys.path.append(os.path.dirname(__file__))
