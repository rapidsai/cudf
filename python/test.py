import cudf


a = cudf.Series([None, None, None], dtype='str')
gdf = cudf.DataFrame()

gdf[0] = cudf.Series([None, None, None], dtype='str')
gdf['a'] = [1, 2, 3]

gdf2 = gdf.copy()

got = gdf.merge(gdf2, on=[0], how='left')
