# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

from polars.polars import _expr_nodes as pl_expr

from cudf_polars.dsl.expressions.boolean import BooleanFunction
from cudf_polars.dsl.expressions.datetime import TemporalFunction
from cudf_polars.dsl.expressions.string import StringFunction


@pytest.fixture(params=[BooleanFunction, StringFunction, TemporalFunction])
def function(request):
    return request.param


def test_function_name_serialization_all_values(function):
    # Test serialization and deserialization for all values of function.Name
    for name in function.Name:
        serialized_name = pickle.dumps(name)
        deserialized_name = pickle.loads(serialized_name)
        assert deserialized_name is name


def test_function_name_invalid(function):
    # Test invalid attribute name
    with pytest.raises(
        AttributeError, match="type object 'Name' has no attribute 'InvalidAttribute'"
    ):
        assert function.Name.InvalidAttribute is function.Name.InvalidAttribute


def test_from_polars_all_names(function):
    # Test that all valid names of polars expressions are correctly converted
    for name in function.Name:
        polars_function = getattr(pl_expr, function.__name__)
        polars_function_attr = getattr(polars_function, name.name)
        cudf_function = function.Name.from_polars(polars_function_attr)
        assert cudf_function == name


def test_from_polars_invalid_attribute(function):
    # Test converting from invalid attribute name
    with pytest.raises(ValueError, match=f"{function.__name__} required"):
        function.Name.from_polars("InvalidAttribute")


def test_from_polars_invalid_polars_attribute(function):
    # Test converting from polars function with invalid attribute name
    with pytest.raises(
        AttributeError, match="type object 'Name' has no attribute 'InvalidAttribute'"
    ):
        function.Name.from_polars(f"{function.__name__}.InvalidAttribute")
