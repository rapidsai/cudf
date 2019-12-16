def reorder_dataframe_columns_to_match_pandas(gdf, pdf):
    """PANDAS>0.25 Column order is preserved when passing a list of dicts to 
    DataFrame. We reorder to match to that.

    https://pandas.pydata.org/pandas-docs/version/0.25/whatsnew/v0.25.0.html#column-order-is-preserved-when-passing-a-list-of-dicts-to-dataframe


    Examples:
    ---------

    Cudf returns column order in the same manner as pandas<0.25.0. 

    In [63]: data = [
    ....:     {'name': 'Joe', 'state': 'NY', 'age': 18},
    ....:     {'name': 'Jane', 'state': 'KY', 'age': 19, 'hobby': 'Minecraft'},
    ....:     {'name': 'Jean', 'state': 'OK', 'age': 20, 'finances': 'good'}
    ....: ]
    ....: 


    In [1]: cudf.DataFrame(data)
    Out[1]:
       age finances      hobby  name state
    0   18      NaN        NaN   Joe    NY
    1   19      NaN  Minecraft  Jane    KY
    2   20     good        NaN  Jean    OK


    In [64]: pd.DataFrame(data)
    Out[64]: 
       name state  age      hobby finances
    0   Joe    NY   18        NaN      NaN
    1  Jane    KY   19  Minecraft      NaN
    2  Jean    OK   20        NaN     good

    [3 rows x 5 columns]

    In [128]: pandasutils.reorder_dataframe_columns_to_match_pandas(gdf=cudf.DataFrame(data), pdf=pd.DataFrame(data))
    Out[128]: 
       name state  age      hobby finances
    0   Joe    NY   18        NaN      NaN
    1  Jane    KY   19  Minecraft      NaN
    2  Jean    OK   20        NaN     good
    
    Parameters
    ----------
    df : cudf.Dataframe
    order : list of fields

    Returns
    -------
    cudf.Dataframe
        cudf Dataframe reordered to match pandas enforced column index naming.
    """

    from cudf import DataFrame as CuDataFrame
    from pandas import DataFrame as PaDataFrame

    if not isinstance(gdf, CuDataFrame) or not isinstance(pdf, PaDataFrame):
        raise AttributeError()
    else:
        return gdf[pdf.columns.tolist()]
