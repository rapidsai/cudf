# Copyright (c) 2022, NVIDIA CORPORATION.

from utils import benchmark_with_object


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False)
def where_case_1(dataframe):
    return dataframe, dataframe % 2 == 0, 0


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False)
def where_case_2(dataframe):
    cond = dataframe[dataframe.columns[0]] % 2 == 0
    return dataframe, cond, 0
