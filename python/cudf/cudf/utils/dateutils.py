# Copyright (c) 2019, NVIDIA CORPORATION.

import calendar
from cudf.utils.dtypes import is_scalar

day_names = list(calendar.day_name)
month_names = list(calendar.month_name)

def wkday_idx_to_name(inds_in):
    indices = range(len(day_names) + 1)
    ind_to_day = dict(zip(indices, day_names))
    if is_scalar(inds_in):
        return ind_to_day[inds_in]
    else:
        return [ind_to_day[i] for i in inds_in]

def month_idx_to_name(inds_in):
    indices = range(len(month_names) + 1)
    ind_to_month = dict(zip(indices, month_names))
    if is_scalar(inds_in):
        return ind_to_month[inds_in]
    else:
        return [ind_to_month[i] for i in inds_in]

