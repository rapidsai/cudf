import pandas as pd
import numpy as np
from cudf import DataFrame
from cudf import Series
from cudf.dataframe import Buffer
from cudf.utils import cudautils
from cudf.dataframe.categorical import CategoricalColumn

# ##### TOPIC: VSCode debug support ==============================================
# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('10.110.44.117', 3000), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()
# # ==============================================================================



# ##### TOPIC: Categorical internal working ======================================
# pdcatser = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
# ser = Series.from_categorical(pdcatser)
# # ==============================================================================



# ##### TOPIC: cuDF string test ==================================================
# pdf = pd.DataFrame({'A': {0: 'a', 1: 'a', 2: 'c'},
#                    'B': {0: 1, 1: 1, 2: 5},
#                    'C': {0: 1, 1: 1, 2: 6},
#                    'D': {0: 1.0, 1: 3.0, 2: 4.0}})
# print(pdf)

# df = DataFrame.from_pandas(pdf)
# # ==============================================================================



# ##### TOPIC: Tiling test =======================================================
# def _tile(A, reps):
#     series_list = []
#     for i in range(0, reps):
#         series_list.append(A)
#     return Series._concat(objs=series_list, index=True)
# cols = {}
# for col in ['C', 'D']:
#     cols[col] = _tile(df[col], 2)

# df2 = DataFrame(cols)
# print(df2)
# # ==============================================================================



# ##### TOPIC: Concat test =======================================================
# # sers = []
# # sers.append(df['B'])
# # sers.append(df['C'])
# #
# # newser = Series._concat(sers)
# # ==============================================================================



##### TOPIC: Basic =============================================================
pdf = pd.DataFrame({'A': {0: 1, 1: 1, 2: 5},
                   'B': {0: 1, 1: 3, 2: 6},
                   'C': {0: 1.0, 1: np.nan, 2: 4.0},
                   'D': {0: 2.0, 1: 5.0, 2: 6.0}})
print(pdf)
# ==============================================================================



##### TOPIC: pandas melt =======================================================
pdf1 = pd.melt(pdf, id_vars=['A','C'], value_vars=['B','D'])
print(pdf1)
# ==============================================================================



##### TOPIC: cuDF melt api =====================================================
from cudf.reshape import melt
df = DataFrame.from_pandas(pdf)
df2 = melt(frame=df, id_vars=['A', 'B'], value_vars=['C', 'D'])
print(df2)
# ==============================================================================



# ##### TOPIC: cuDF melt scratch =================================================
# from numba import cuda
# from librmm_cffi import librmm as rmm


# # Args ========
# frame = DataFrame.from_pandas(pdf)
# id_vars = ['A', 'C']
# value_vars = ['B', 'D']
# var_name = 'variable'
# value_name = 'value'
# # =============

# # Arg cleaning
# import types
# # id_vars
# if id_vars is not None:
#     if not isinstance(id_vars, list):
#         id_vars = [id_vars]
#     missing = set(id_vars) - set(frame.columns)
#     if not len(missing) == 0:
#         raise KeyError(
#             "The following 'id_vars' are not present"
#             " in the DataFrame: {missing}"
#             "".format(missing=list(missing)))
# else:
#     id_vars = []

# # value_vars
# if value_vars is not None:
#     if not isinstance(value_vars, list):
#         value_vars = [value_vars]
#     missing = set(value_vars) - set(frame.columns)
#     if not len(missing) == 0:
#         raise KeyError(
#             "The following 'value_vars' are not present"
#             " in the DataFrame: {missing}"
#             "".format(missing=list(missing)))
# else:
#     value_vars = []

# # overlap
# overlap = set(id_vars).intersection(set(value_vars))
# if not len(overlap) == 0:
#     raise KeyError(
#         "'value_vars' and 'id_vars' cannot have overlap."
#         " The following 'value_vars' are ALSO present"
#         " in 'id_vars': {overlap}"
#         "".format(overlap=list(overlap)))

# N = len(frame)
# K = len(value_vars)

# def _tile(A, reps):
#     series_list = [A] * reps
#     return Series._concat(objs=series_list, index=None)

# # Step 1: tile id_vars
# mdata = {}
# for col in id_vars:
#     mdata[col] = _tile(frame[col], K)
    
# # Step 2: add variable
# var_cols = []
# for i, var in enumerate(value_vars):
#     var_cols.append(Series(Buffer(
#         cudautils.full(size=N, value=i, dtype=np.int8))))
# temp = Series._concat(objs=var_cols, index=None)
# mdata[var_name] = Series(CategoricalColumn(
#     categories=tuple(value_vars), data=temp._column.data, ordered=False))

# # Step 3: add values
# mdata[value_name] = Series._concat(
#     objs=[frame[val] for val in value_vars],
#     index=None)

# print(DataFrame(mdata))
# # ==============================================================================



# ##### TOPIC: is from_pandas even working =======================================
# pdf = pd.DataFrame({'a': [np.nan, 1.0, 2.0, 3.0, 1, 1, 1, 1],
#     'b': [4, 5, np.nan, 7, 1, 1, 1, 1],
#     'c': [np.nan, 9.0, np.nan, 11.0, 1, 1, 1, 1],
#     'd': [4, 5, np.nan, 7, 1, 1, 1, 1],
#     'e': [4, 5, np.nan, 7, 1, 1, 1, np.nan],
#     'f': [4, 5, np.nan, 7, 1, 1, 1, 1],
#     'g': [4, 5, 6, 7, 1, np.nan, 1, 1],
#     'h': [4, 5, 6.0, 7, 1, 1, 1, 1],
#     'j': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#     'k': [7, 1.0, 2.0, 3.0, 1, 1, 1, 1]})
# df = DataFrame.from_pandas(pdf)
# # ==============================================================================

pass