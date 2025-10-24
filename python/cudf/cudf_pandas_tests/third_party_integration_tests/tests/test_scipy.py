# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import scipy


@pytest.mark.parametrize("func", ["hmean", "tvar", "gstd"])
def test_scipy_stats(func):
    rng = np.random.default_rng(42)
    data = pd.Series(rng.random(1000))
    return getattr(scipy.stats, func)(data)


@pytest.mark.parametrize("func", ["norm"])
def test_scipy_linalg(func):
    rng = np.random.default_rng(42)
    data = pd.Series(rng.random(1000))
    return getattr(scipy.linalg, func)(data)


pytestmark = pytest.mark.assert_eq(fn=pd._testing.assert_almost_equal)


def test_compute_pi():
    def circle(x):
        return (1 - x**2) ** 0.5

    x = pd.Series(np.linspace(0, 1, 100))
    y = pd.Series(circle(np.linspace(0, 1, 100)))

    result = scipy.integrate.trapezoid(y, x)
    return result * 4


def test_matrix_solve():
    A = pd.DataFrame([[2, 3], [1, 2]])
    b = pd.Series([1, 2])

    return scipy.linalg.solve(A, b)


def test_correlation():
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

    return scipy.stats.pearsonr(data["A"], data["B"])


def test_optimization():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    def rosen(x):  # banana function from scipy tutorial
        return sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )

    result = scipy.optimize.fmin(rosen, x)
    return result


def test_regression():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    result = scipy.stats.linregress(data["y"], data["y"])
    return result.slope, result.intercept
