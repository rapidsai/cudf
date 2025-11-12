# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cugraph
import cupy as cp
import networkx as nx
import numpy as np
import pandas as pd
import pytest

cugraph_algos = [
    "betweenness_centrality",
    "degree_centrality",
    "katz_centrality",
    "sorensen_coefficient",
    "jaccard_coefficient",
]

nx_algos = [
    "betweenness_centrality",
    "degree_centrality",
    "katz_centrality",
]


def assert_cugraph_equal(expect, got):
    if isinstance(expect, cp.ndarray):
        expect = expect.get()
    if isinstance(got, cp.ndarray):
        got = got.get()
    elif isinstance(expect, np.ndarray) and isinstance(got, np.ndarray):
        assert np.array_equal(expect, got)
    else:
        assert expect == got


pytestmark = pytest.mark.assert_eq(fn=assert_cugraph_equal)


@pytest.fixture(scope="session")
def df():
    return pd.DataFrame({"source": [0, 1, 2], "destination": [1, 2, 3]})


@pytest.fixture(scope="session")
def adjacency_matrix():
    data = {
        1: [0, 1, 1, 0],
        2: [1, 0, 0, 1],
        3: [1, 0, 0, 1],
        4: [0, 1, 1, 0],
    }
    df = pd.DataFrame(data, index=[1, 2, 3, 4])
    return df


@pytest.mark.parametrize("algo", cugraph_algos)
def test_cugraph_from_pandas_edgelist(df, algo):
    G = cugraph.Graph()
    G.from_pandas_edgelist(df)
    return getattr(cugraph, algo)(G).to_pandas().values


@pytest.mark.parametrize("algo", cugraph_algos)
def test_cugraph_from_pandas_adjacency(adjacency_matrix, algo):
    G = cugraph.Graph()
    G.from_pandas_adjacency(adjacency_matrix)
    res = getattr(cugraph, algo)(G).to_pandas()
    return res.sort_values(list(res.columns)).values


@pytest.mark.parametrize("algo", cugraph_algos)
def test_cugraph_from_numpy_array(df, algo):
    G = cugraph.Graph()
    G.from_numpy_array(df.values)
    return getattr(cugraph, algo)(G).to_pandas().values


@pytest.mark.parametrize("algo", nx_algos)
def test_networkx_from_pandas_edgelist(df, algo):
    G = nx.from_pandas_edgelist(
        df, "source", "destination", ["source", "destination"]
    )
    return getattr(nx, algo)(G)


@pytest.mark.parametrize("algo", nx_algos)
def test_networkx_from_pandas_adjacency(adjacency_matrix, algo):
    G = nx.from_pandas_adjacency(adjacency_matrix)
    return getattr(nx, algo)(G)


@pytest.mark.parametrize("algo", nx_algos)
def test_networkx_from_numpy_array(adjacency_matrix, algo):
    G = nx.from_numpy_array(adjacency_matrix.values)
    return getattr(nx, algo)(G)
