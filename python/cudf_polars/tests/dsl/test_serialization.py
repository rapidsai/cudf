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
    with pytest.raises(AttributeError, match="InvalidAttribute"):
        assert function.Name.InvalidAttribute is function.Name.InvalidAttribute


def test_from_polars_all_names(function):
    # Test that all valid names of polars expressions are correctly converted
    polars_function = getattr(pl_expr, function.__name__)
    polars_names = [name for name in dir(polars_function) if not name.startswith("_")]
    # Check names advertised by polars are the same as we advertise
    assert set(polars_names) == set(function.Name.__members__)
    for name in function.Name:
        attr = getattr(polars_function, name.name)
        assert function.Name.from_polars(attr) == name


def test_from_polars_invalid_attribute(function):
    # Test converting from invalid attribute name
    with pytest.raises(ValueError, match=f"{function.__name__} required"):
        function.Name.from_polars("InvalidAttribute")


def test_from_polars_invalid_polars_attribute(function):
    # Test converting from polars function with invalid attribute name
    with pytest.raises(AttributeError, match="InvalidAttribute"):
        function.Name.from_polars(f"{function.__name__}.InvalidAttribute")
