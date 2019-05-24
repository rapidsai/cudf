import pytest

import numpy as np
import pandas as pd
from numba import cuda

from cudf.dataframe.dataframe import DataFrame
from cudf.tests.utils import assert_eq

def assert_df(g, p):
    assert g.index.dtype == p.index.dtype
    np.testing.assert_equal(g.index.to_array(), p.index)
    assert tuple(g.columns) == tuple(p.columns)
    for k in g.columns:
        assert g[k].dtype == p[k].dtype
        np.testing.assert_equal(g[k].to_array(), p[k])



@pytest.mark.parametrize('subset', ['a', ['a'], ['a', 'B']])
def test_duplicated_with_misspelled_column_name(subset):
    df = pd.DataFrame({'A': [0, 0, 1],
                    'B': [0, 0, 1],
                    'C': [0, 0, 1]})
    df = DataFrame.from_pandas(df)

    with pytest.raises(KeyError):
        df.drop_duplicates(subset)


@pytest.mark.parametrize('df', [
    pd.DataFrame(),
    pd.DataFrame(columns=[]),
    pd.DataFrame(columns=['A', 'B', 'C']),
    pd.DataFrame(index=[]),
#    pd.DataFrame(index=['A', 'B', 'C'])
])
def test_drop_duplicates_empty(df):
    df = DataFrame.from_pandas(df)
    result = df.drop_duplicates()
    assert_eq(result, df)

    result = df.copy()
    result.drop_duplicates(inplace=True)
    assert_eq(result, df)


@pytest.mark.parametrize('num_columns', [3, 4, 5])
def test_dataframe_drop_duplicates_numeric_method(num_columns):
    import random
    import itertools as it
    comb=list(it.permutations(range(num_columns), num_columns))
    shuf=list(comb)
    random.Random(num_columns).shuffle(shuf)

    #create dataframe with n_dup duplicate rows
    def get_pdf(n_dup):
        rows = comb + shuf[:n_dup]
        random.Random(n_dup).shuffle(rows)
        return pd.DataFrame(rows)

    for i in range(5):
        pdf = get_pdf(i)
        gdf = DataFrame.from_pandas(pdf)
        assert_eq(gdf.drop_duplicates() , pdf.drop_duplicates())

    # subset columns, single columns
    assert_eq(gdf.drop_duplicates(pdf.columns[:-1]) , pdf.drop_duplicates(pdf.columns[:-1]))
    assert_eq(gdf.drop_duplicates(pdf.columns[-1]) , pdf.drop_duplicates(pdf.columns[-1]))
    assert_eq(gdf.drop_duplicates(pdf.columns[0]) , pdf.drop_duplicates(pdf.columns[0]))

    # subset columns shuffled
    cols =  list(pdf.columns)
    random.Random(3).shuffle(cols)
    assert_eq(gdf.drop_duplicates(cols) , pdf.drop_duplicates(cols))
    random.Random(3).shuffle(cols)
    assert_eq(gdf.drop_duplicates(cols[:-1]) , pdf.drop_duplicates(cols[:-1]))
    random.Random(3).shuffle(cols)
    assert_eq(gdf.drop_duplicates(cols[-1]) , pdf.drop_duplicates(cols[-1]))
    assert_eq(gdf.drop_duplicates(cols, keep='last') , pdf.drop_duplicates(cols, keep='last'))


@pytest.mark.skip(reason="string column unsupported yet (issue #1467)")
def test_dataframe_drop_duplicates_method():
    pdf = pd.DataFrame([(1, 2, 'a'),
                        (2, 3, 'b'),
                        (3, 4, 'c'),
                        (2, 3, 'd'),
                        (3, 5, 'c')], columns=['n1', 'n2', 's1'])
    gdf = DataFrame.from_pandas(pdf)

    assert tuple(gdf.drop_duplicates('n1')['n1']) == (1, 2, 3)
    assert tuple(gdf.drop_duplicates('n2')['n2']) == (2, 3, 4, 5)
    assert tuple(gdf.drop_duplicates('s1')['s1']) == ('a', 'b', 'c', 'd')
    assert tuple(gdf.drop_duplicates('s1', keep='last')['s1']) == ('a', 'b', 'd', 'c')
    assert gdf.drop_duplicates('s1', inplace=True) == None

    def assert_df(g, p):
        assert g.index.dtype == p.index.dtype
        np.testing.assert_equal(g.index.to_array(), p.index)
        assert tuple(g.columns) == tuple(p.columns)
        for k in g.columns:
            assert g[k].dtype == p[k].dtype
            np.testing.assert_equal(g[k].to_array(), p[k])

    assert_eq(gdf.drop_duplicates('n1') , pdf.drop_duplicates('n1'))
    assert_eq(gdf.drop_duplicates('n2') , pdf.drop_duplicates('n2'))
    assert_eq(gdf.drop_duplicates('s1') , pdf.drop_duplicates('s1'))
    assert_eq(gdf.drop_duplicates(['n1', 'n2']) , pdf.drop_duplicates(['n1', 'n2']))
    assert_eq(gdf.drop_duplicates(['n1', 's1']) , pdf.drop_duplicates(['n1', 's1']))

    # Test drop error
    with pytest.raises(NameError) as raises:
        df.drop_duplicates('n3')
    raises.match("column 'n3' does not exist")
    with pytest.raises(NameError) as raises:
        df.drop(['n1', 'n3', 'n2'])
    raises.match("column 'n3' does not exist")
