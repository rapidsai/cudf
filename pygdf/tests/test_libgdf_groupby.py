# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def make_frame(dataframe_class, nelem, seed=0, extra_levels=()):
    np.random.seed(seed)

    df = dataframe_class()

    df['x'] = np.random.randint(0, 5, nelem)
    df['y'] = np.random.randint(0, 3, nelem)
    for lvl in extra_levels:
        df[lvl] = np.random.randint(0, 2, nelem)

    df['val'] = np.random.random(nelem)

    return df


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean(nelem):
    # gdf
    got_df = make_frame(DataFrame, nelem=nelem).groupby(('x', 'y'), method = "GDF_SORT").mean()
    got = np.sort(got_df['val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame,
                           nelem=nelem).groupby(('x', 'y')).mean()
    expect = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect, got)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean_3level(nelem):
    lvls = 'z'
    bys = list('xyz')
    # gdf
    got_df = make_frame(DataFrame, nelem=nelem,
                        extra_levels=lvls).groupby(bys, method = "GDF_SORT").mean()
    got = np.sort(got_df['val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem,
                           extra_levels=lvls).groupby(bys).mean()
    expect = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect, got)


@pytest.mark.parametrize('nelem', [2, 100])
def test_groupby_agg_mean_min(nelem):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem).groupby(('x', 'y'), method = "GDF_SORT")\
                                               .agg(['mean', 'min'])
    got_mean = np.sort(got_df['val_mean'].to_array())
    got_min = np.sort(got_df['val_min'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(('x', 'y'))\
                                                     .agg(['mean', 'min'])
    expect_mean = np.sort(expect_df['val', 'mean'].values)
    expect_min = np.sort(expect_df['val', 'min'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_mean, got_mean)
    np.testing.assert_array_almost_equal(expect_min, got_min)

# 
# def test_groupby_cats():
#     df = DataFrame()
#     df['cats'] = pd.Categorical(list('aabaacaab'))
#     df['vals'] = np.random.random(len(df))
# 
#     cats = np.asarray(list(df['cats']))
#     vals = df['vals'].to_array()
# 
#     grouped = df.groupby(['cats'], method = "pygdf").mean()
# 
#     got_vals = grouped['vals']
#     got_cats = grouped['cats']
# 
#     for c, v in zip(got_cats, got_vals):
#         print(c, v)
#         expect = vals[cats == c].mean()
#         np.testing.assert_almost_equal(v, expect)
# 
# 
# def test_groupby_iterate_groups():
#     np.random.seed(0)
#     df = DataFrame()
#     nelem = 20
#     df['key1'] = np.random.randint(0, 3, nelem)
#     df['key2'] = np.random.randint(0, 2, nelem)
#     df['val1'] = np.random.random(nelem)
#     df['val2'] = np.random.random(nelem)
# 
#     def assert_values_equal(arr):
#         np.testing.assert_array_equal(arr[0], arr)
# 
#     for grp in df.groupby(['key1', 'key2'], method = "pygdf"):
#         pddf = grp.to_pandas()
#         for k in 'key1,key2'.split(','):
#             assert_values_equal(pddf[k].values)
# 
# 
# def test_groupby_as_df():
#     np.random.seed(0)
#     df = DataFrame()
#     nelem = 20
#     df['key1'] = np.random.randint(0, 3, nelem)
#     df['key2'] = np.random.randint(0, 2, nelem)
#     df['val1'] = np.random.random(nelem)
#     df['val2'] = np.random.random(nelem)
# 
#     def assert_values_equal(arr):
#         np.testing.assert_array_equal(arr[0], arr)
# 
#     df, segs = df.groupby(['key1', 'key2'], method = "pygdf").as_df()
#     for s, e in zip(segs, list(segs[1:]) + [None]):
#         grp = df[s:e]
#         pddf = grp.to_pandas()
#         for k in 'key1,key2'.split(','):
#             assert_values_equal(pddf[k].values)
# 
# 
# def test_groupby_apply():
#     np.random.seed(0)
#     df = DataFrame()
#     nelem = 20
#     df['key1'] = np.random.randint(0, 3, nelem)
#     df['key2'] = np.random.randint(0, 2, nelem)
#     df['val1'] = np.random.random(nelem)
#     df['val2'] = np.random.random(nelem)
# 
#     expect_grpby = df.to_pandas().groupby(['key1', 'key2'],
#                                           as_index=False)
#     got_grpby = df.groupby(['key1', 'key2'], method = "pygdf")
# 
#     def foo(df):
#         df['out'] = df['val1'] + df['val2']
#         return df
# 
#     expect = expect_grpby.apply(foo)
#     expect = expect.sort_values(['key1', 'key2']).reset_index(drop=True)
# 
#     got = got_grpby.apply(foo).to_pandas()
#     pd.util.testing.assert_frame_equal(expect, got)
# 
# 
# def test_groupby_apply_grouped():
#     from numba import cuda
# 
#     np.random.seed(0)
#     df = DataFrame()
#     nelem = 20
#     df['key1'] = np.random.randint(0, 3, nelem)
#     df['key2'] = np.random.randint(0, 2, nelem)
#     df['val1'] = np.random.random(nelem)
#     df['val2'] = np.random.random(nelem)
# 
#     expect_grpby = df.to_pandas().groupby(['key1', 'key2'], as_index=False)
#     got_grpby = df.groupby(['key1', 'key2'], method = "pygdf")
# 
#     def foo(key1, val1, com1, com2):
#         for i in range(cuda.threadIdx.x, len(key1), cuda.blockDim.x):
#             com1[i] = key1[i] * 10000 + val1[i]
#             com2[i] = i
# 
#     got = got_grpby.apply_grouped(foo,
#                                   incols=['key1', 'val1'],
#                                   outcols={'com1': np.float64,
#                                            'com2': np.int32},
#                                   tpb=8)
# 
#     got = got.to_pandas()
# 
#     # Get expected result by emulating the operation in pandas
#     def emulate(df):
#         df['com1'] = df.key1 * 10000 + df.val1
#         df['com2'] = np.arange(len(df), dtype=np.int32)
#         return df
#     expect = expect_grpby.apply(emulate)
#     expect = expect.sort_values(['key1', 'key2']).reset_index(drop=True)
# 
#     pd.util.testing.assert_frame_equal(expect, got)
# 
# 
# @pytest.mark.parametrize('nelem', [100, 500])
# @pytest.mark.parametrize('func', ['mean', 'std', 'var', 'min',
#                                   'max', 'count', 'sum'])
# def test_groupby_2keys_agg(nelem, func):
#     # gdf (Note: lack of multindex)
#     got_df = make_frame(DataFrame, nelem=nelem)\
#         .groupby(('x', 'y'), method = "pygdf").agg(func)
# 
#     got_agg = np.sort(got_df['val'].to_array())
#     # pandas
#     expect_df = make_frame(pd.DataFrame, nelem=nelem)\
#         .groupby(('x', 'y')).agg(func)
# 
#     expect_agg = np.sort(expect_df['val'].values)
#     # verify
#     np.testing.assert_array_almost_equal(expect_agg, got_agg)
