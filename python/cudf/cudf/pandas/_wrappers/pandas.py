# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import copyreg
import importlib
import os
import pickle
import sys

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar as pd_AbstractHolidayCalendar,
    EasterMonday as pd_EasterMonday,
    GoodFriday as pd_GoodFriday,
    Holiday as pd_Holiday,
    HolidayCalendarFactory as pd_HolidayCalendarFactory,
    HolidayCalendarMetaClass as pd_HolidayCalendarMetaClass,
    USColumbusDay as pd_USColumbusDay,
    USFederalHolidayCalendar as pd_USFederalHolidayCalendar,
    USLaborDay as pd_USLaborDay,
    USMartinLutherKingJr as pd_USMartinLutherKingJr,
    USMemorialDay as pd_USMemorialDay,
    USPresidentsDay as pd_USPresidentsDay,
    USThanksgivingDay as pd_USThanksgivingDay,
)

import cudf
import cudf.core._compat

from ..annotation import nvtx
from ..fast_slow_proxy import (
    _CUDF_PANDAS_NVTX_COLORS,
    _DELETE,
    _fast_slow_function_call,
    _FastSlowAttribute,
    _FunctionProxy,
    _Unusable,
    make_final_proxy_type as _make_final_proxy_type,
    make_intermediate_proxy_type as _make_intermediate_proxy_type,
    register_proxy_func,
)
from .common import (
    array_function_method,
    array_method,
    arrow_array_method,
    cuda_array_interface,
    custom_iter,
)

from pandas.io.sas.sas7bdat import (  # isort: skip
    SAS7BDATReader as pd_SAS7BDATReader,
)
from pandas.io.sas.sas_xport import (  # isort: skip
    XportReader as pd_XportReader,
)

# TODO(pandas2.1): Can import from pandas.api.typing
from pandas.core.resample import (  # isort: skip
    Resampler as pd_Resampler,
    TimeGrouper as pd_TimeGrouper,
)

try:
    from IPython import get_ipython

    ipython_shell = get_ipython()
except ImportError:
    ipython_shell = None

cudf.set_option("mode.pandas_compatible", True)


def _pandas_util_dir():
    # In pandas 2.0, pandas.util contains public APIs under
    # __getattr__ but no __dir__ to find them
    # https://github.com/pandas-dev/pandas/blob/2.2.x/pandas/util/__init__.py
    res = list(
        set(
            [
                *list(importlib.import_module("pandas.util").__dict__.keys()),
                "Appender",
                "Substitution",
                "_exceptions",
                "_print_versions",
                "cache_readonly",
                "hash_array",
                "hash_pandas_object",
                "version",
                "_tester",
                "_validators",
                "_decorators",
            ]
        )
    )
    if cudf.core._compat.PANDAS_GE_220:
        res.append("capitalize_first_letter")
    return res


pd.util.__dir__ = _pandas_util_dir


def make_final_proxy_type(
    name,
    fast_type,
    slow_type,
    **kwargs,
):
    assert "module" not in kwargs
    return _make_final_proxy_type(
        name, fast_type, slow_type, module=slow_type.__module__, **kwargs
    )


def make_intermediate_proxy_type(name, fast_type, slow_type):
    return _make_intermediate_proxy_type(
        name, fast_type, slow_type, module=slow_type.__module__
    )


class _AccessorAttr:
    """
    Descriptor that ensures that accessors like `.dt` and `.str`
    return the corresponding accessor types when accessed on `Series`
    and `Index` _types_ (not instances).n

    Attribute access for _instances_ uses the regular fast-then-slow
    lookup defined in `__getattr__`.
    """

    def __init__(self, typ):
        self._typ = typ

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, cls=None):
        if obj is None:
            return self._typ
        else:
            return _FastSlowAttribute(self._name).__get__(obj, type(obj))


def Timestamp_Timedelta__new__(cls, *args, **kwargs):
    # Call fast/slow constructor
    # This takes care of running __init__ as well, but must be paired
    # with a removal of the defaulted __init__ that
    # make_final_proxy_type provides.
    # Timestamp & Timedelta don't always return same types as self,
    # hence this method is needed.
    self, _ = _fast_slow_function_call(
        lambda cls, args, kwargs: cls(*args, **kwargs),
        cls,
        args,
        kwargs,
    )
    return self


Timedelta = make_final_proxy_type(
    "Timedelta",
    _Unusable,
    pd.Timedelta,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__new__": Timestamp_Timedelta__new__,
        "__init__": _DELETE,
    },
)


Timestamp = make_final_proxy_type(
    "Timestamp",
    _Unusable,
    pd.Timestamp,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__new__": Timestamp_Timedelta__new__,
        "__init__": _DELETE,
    },
)

DatetimeProperties = make_intermediate_proxy_type(
    "DatetimeProperties",
    cudf.core.series.DatetimeProperties,
    pd.core.indexes.accessors.DatetimeProperties,
)

TimedeltaProperties = make_intermediate_proxy_type(
    "TimedeltaProperties",
    cudf.core.series.TimedeltaProperties,
    pd.core.indexes.accessors.TimedeltaProperties,
)

CombinedDatetimelikeProperties = make_intermediate_proxy_type(
    "CombinedDatetimelikeProperties",
    cudf.core.series.DatetimeProperties,
    pd.core.indexes.accessors.CombinedDatetimelikeProperties,
)

StringMethods = make_intermediate_proxy_type(
    "StringMethods",
    cudf.core.column.string.StringMethods,
    pd.core.strings.accessor.StringMethods,
)

_CategoricalAccessor = make_intermediate_proxy_type(
    "CategoricalAccessor",
    cudf.core.column.categorical.CategoricalAccessor,
    pd.core.arrays.categorical.CategoricalAccessor,
)


def _DataFrame__dir__(self):
    # Column names that are string identifiers are added to the dir of the
    # DataFrame
    # See https://github.com/pandas-dev/pandas/blob/43691a2f5d235b08f0f3aa813d8fdcb7c4ce1e47/pandas/core/indexes/base.py#L878
    _pd_df_dir = dir(pd.DataFrame)
    return _pd_df_dir + [
        colname
        for colname in self.columns
        if isinstance(colname, str) and colname.isidentifier()
    ]


def ignore_ipython_canary_check(self, **kwargs):
    raise AttributeError(
        "_ipython_canary_method_should_not_exist_ doesn't exist"
    )


DataFrame = make_final_proxy_type(
    "DataFrame",
    cudf.DataFrame,
    pd.DataFrame,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__dir__": _DataFrame__dir__,
        "_constructor": _FastSlowAttribute("_constructor"),
        "_constructor_sliced": _FastSlowAttribute("_constructor_sliced"),
        "_accessors": set(),
        "_ipython_canary_method_should_not_exist_": ignore_ipython_canary_check,
    },
)


def custom_repr_html(obj):
    # This custom method is need to register a html format
    # for ipython
    return _fast_slow_function_call(
        lambda obj: obj._repr_html_(),
        obj,
    )[0]


if ipython_shell:
    # See: https://ipython.readthedocs.io/en/stable/config/integrating.html#formatters-for-third-party-types
    html_formatter = ipython_shell.display_formatter.formatters["text/html"]
    html_formatter.for_type(DataFrame, custom_repr_html)


Series = make_final_proxy_type(
    "Series",
    cudf.Series,
    pd.Series,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__array_function__": array_function_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        "__iter__": custom_iter,
        "dt": _AccessorAttr(CombinedDatetimelikeProperties),
        "str": _AccessorAttr(StringMethods),
        "cat": _AccessorAttr(_CategoricalAccessor),
        "_constructor": _FastSlowAttribute("_constructor"),
        "_constructor_expanddim": _FastSlowAttribute("_constructor_expanddim"),
        "_accessors": set(),
    },
)


def Index__new__(cls, *args, **kwargs):
    # Call fast/slow constructor
    # This takes care of running __init__ as well, but must be paired
    # with a removal of the defaulted __init__ that
    # make_final_proxy_type provides.
    self, _ = _fast_slow_function_call(
        lambda cls, args, kwargs: cls(*args, **kwargs),
        cls,
        args,
        kwargs,
    )
    return self


def Index__setattr__(self, name, value):
    if name.startswith("_"):
        object.__setattr__(self, name, value)
        return
    if name == "name":
        setattr(self._fsproxy_wrapped, "name", value)
    if name == "names":
        setattr(self._fsproxy_wrapped, "names", value)
    return _FastSlowAttribute("__setattr__").__get__(self, type(self))(
        name, value
    )


Index = make_final_proxy_type(
    "Index",
    cudf.Index,
    pd.Index,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__array_function__": array_function_method,
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        "dt": _AccessorAttr(CombinedDatetimelikeProperties),
        "str": _AccessorAttr(StringMethods),
        "cat": _AccessorAttr(_CategoricalAccessor),
        "__iter__": custom_iter,
        "__init__": _DELETE,
        "__new__": Index__new__,
        "__setattr__": Index__setattr__,
        "_constructor": _FastSlowAttribute("_constructor"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_accessors": set(),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

RangeIndex = make_final_proxy_type(
    "RangeIndex",
    cudf.RangeIndex,
    pd.RangeIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "name": _FastSlowAttribute("name"),
    },
)

SparseDtype = make_final_proxy_type(
    "SparseDtype",
    _Unusable,
    pd.SparseDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

SparseArray = make_final_proxy_type(
    "SparseDtype",
    _Unusable,
    pd.arrays.SparseArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

CategoricalIndex = make_final_proxy_type(
    "CategoricalIndex",
    cudf.CategoricalIndex,
    pd.CategoricalIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "name": _FastSlowAttribute("name"),
    },
)

Categorical = make_final_proxy_type(
    "Categorical",
    _Unusable,
    pd.Categorical,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

CategoricalDtype = make_final_proxy_type(
    "CategoricalDtype",
    cudf.CategoricalDtype,
    pd.CategoricalDtype,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

DatetimeIndex = make_final_proxy_type(
    "DatetimeIndex",
    cudf.DatetimeIndex,
    pd.DatetimeIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

DatetimeArray = make_final_proxy_type(
    "DatetimeArray",
    _Unusable,
    pd.arrays.DatetimeArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

DatetimeTZDtype = make_final_proxy_type(
    "DatetimeTZDtype",
    _Unusable,
    pd.DatetimeTZDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

TimedeltaIndex = make_final_proxy_type(
    "TimedeltaIndex",
    cudf.TimedeltaIndex,
    pd.TimedeltaIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

try:
    from pandas.arrays import NumpyExtensionArray as pd_NumpyExtensionArray

    NumpyExtensionArray = make_final_proxy_type(
        "NumpyExtensionArray",
        _Unusable,
        pd_NumpyExtensionArray,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "_ndarray": _FastSlowAttribute("_ndarray"),
            "_dtype": _FastSlowAttribute("_dtype"),
        },
    )

except ImportError:
    from pandas.arrays import PandasArray as pd_PandasArray

    PandasArray = make_final_proxy_type(
        "PandasArray",
        _Unusable,
        pd_PandasArray,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "_ndarray": _FastSlowAttribute("_ndarray"),
            "_dtype": _FastSlowAttribute("_dtype"),
        },
    )

TimedeltaArray = make_final_proxy_type(
    "TimedeltaArray",
    _Unusable,
    pd.arrays.TimedeltaArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

PeriodIndex = make_final_proxy_type(
    "PeriodIndex",
    _Unusable,
    pd.PeriodIndex,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

PeriodArray = make_final_proxy_type(
    "PeriodArray",
    _Unusable,
    pd.arrays.PeriodArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

PeriodDtype = make_final_proxy_type(
    "PeriodDtype",
    _Unusable,
    pd.PeriodDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

Period = make_final_proxy_type(
    "Period",
    _Unusable,
    pd.Period,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)


MultiIndex = make_final_proxy_type(
    "MultiIndex",
    cudf.MultiIndex,
    pd.MultiIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "names": _FastSlowAttribute("names"),
    },
)

TimeGrouper = make_intermediate_proxy_type(
    "TimeGrouper",
    _Unusable,
    pd_TimeGrouper,
)

Grouper = make_final_proxy_type(
    "Grouper",
    cudf.Grouper,
    pd.Grouper,
    fast_to_slow=lambda fast: pd.Grouper(
        **{
            k: getattr(fast, k)
            for k in {"key", "level", "freq", "closed", "label"}
            if getattr(fast, k) is not None
        }
    ),
    slow_to_fast=lambda slow: cudf.Grouper(
        **{
            k: getattr(slow, k)
            for k in {"key", "level", "freq", "closed", "label"}
            if getattr(slow, k) is not None
        }
    ),
)

StringArray = make_final_proxy_type(
    "StringArray",
    _Unusable,
    pd.arrays.StringArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

if cudf.core._compat.PANDAS_GE_210:
    ArrowStringArrayNumpySemantics = make_final_proxy_type(
        "ArrowStringArrayNumpySemantics",
        _Unusable,
        pd.core.arrays.string_arrow.ArrowStringArrayNumpySemantics,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
    )

ArrowStringArray = make_final_proxy_type(
    "ArrowStringArray",
    _Unusable,
    pd.core.arrays.string_arrow.ArrowStringArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

StringDtype = make_final_proxy_type(
    "StringDtype",
    _Unusable,
    pd.StringDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "storage": _FastSlowAttribute("storage"),
    },
)

BooleanArray = make_final_proxy_type(
    "BooleanArray",
    _Unusable,
    pd.arrays.BooleanArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

BooleanDtype = make_final_proxy_type(
    "BooleanDtype",
    _Unusable,
    pd.BooleanDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

IntegerArray = make_final_proxy_type(
    "IntegerArray",
    _Unusable,
    pd.arrays.IntegerArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

Int8Dtype = make_final_proxy_type(
    "Int8Dtype",
    _Unusable,
    pd.Int8Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)


Int16Dtype = make_final_proxy_type(
    "Int16Dtype",
    _Unusable,
    pd.Int16Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Int32Dtype = make_final_proxy_type(
    "Int32Dtype",
    _Unusable,
    pd.Int32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Int64Dtype = make_final_proxy_type(
    "Int64Dtype",
    _Unusable,
    pd.Int64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

UInt8Dtype = make_final_proxy_type(
    "UInt8Dtype",
    _Unusable,
    pd.UInt8Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

UInt16Dtype = make_final_proxy_type(
    "UInt16Dtype",
    _Unusable,
    pd.UInt16Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

UInt32Dtype = make_final_proxy_type(
    "UInt32Dtype",
    _Unusable,
    pd.UInt32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

UInt64Dtype = make_final_proxy_type(
    "UInt64Dtype",
    _Unusable,
    pd.UInt64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

IntervalIndex = make_final_proxy_type(
    "IntervalIndex",
    cudf.IntervalIndex,
    pd.IntervalIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

IntervalArray = make_final_proxy_type(
    "IntervalArray",
    _Unusable,
    pd.arrays.IntervalArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

IntervalDtype = make_final_proxy_type(
    "IntervalDtype",
    cudf.IntervalDtype,
    pd.IntervalDtype,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Interval = make_final_proxy_type(
    "Interval",
    _Unusable,
    pd.Interval,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

FloatingArray = make_final_proxy_type(
    "FloatingArray",
    _Unusable,
    pd.arrays.FloatingArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
    },
)

Float32Dtype = make_final_proxy_type(
    "Float32Dtype",
    _Unusable,
    pd.Float32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Float64Dtype = make_final_proxy_type(
    "Float64Dtype",
    _Unusable,
    pd.Float64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

SeriesGroupBy = make_intermediate_proxy_type(
    "SeriesGroupBy",
    cudf.core.groupby.groupby.SeriesGroupBy,
    pd.core.groupby.SeriesGroupBy,
)

DataFrameGroupBy = make_intermediate_proxy_type(
    "DataFrameGroupBy",
    cudf.core.groupby.groupby.DataFrameGroupBy,
    pd.core.groupby.DataFrameGroupBy,
)

RollingGroupBy = make_intermediate_proxy_type(
    "RollingGroupBy",
    cudf.core.window.rolling.RollingGroupby,
    pd.core.window.rolling.RollingGroupby,
)

_SeriesIlocIndexer = make_intermediate_proxy_type(
    "_SeriesIlocIndexer",
    cudf.core.series._SeriesIlocIndexer,
    pd.core.indexing._iLocIndexer,
)

_DataFrameIlocIndexer = make_intermediate_proxy_type(
    "_SeriesIlocIndexer",
    cudf.core.dataframe._DataFrameIlocIndexer,
    pd.core.indexing._iLocIndexer,
)

_SeriesLocIndexer = make_intermediate_proxy_type(
    "_SeriesLocIndexer",
    cudf.core.series._SeriesLocIndexer,
    pd.core.indexing._LocIndexer,
)

_DataFrameLocIndexer = make_intermediate_proxy_type(
    "_DataFrameLocIndexer",
    cudf.core.dataframe._DataFrameLocIndexer,
    pd.core.indexing._LocIndexer,
)

_AtIndexer = make_intermediate_proxy_type(
    "_AtIndexer",
    cudf.core.dataframe._DataFrameAtIndexer,
    pd.core.indexing._AtIndexer,
)

_iAtIndexer = make_intermediate_proxy_type(
    "_iAtIndexer",
    cudf.core.dataframe._DataFrameiAtIndexer,
    pd.core.indexing._iAtIndexer,
)

FixedForwardWindowIndexer = make_final_proxy_type(
    "FixedForwardWindowIndexer",
    _Unusable,
    pd.api.indexers.FixedForwardWindowIndexer,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

VariableOffsetWindowIndexer = make_final_proxy_type(
    "VariableOffsetWindowIndexer",
    _Unusable,
    pd.api.indexers.VariableOffsetWindowIndexer,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

Window = make_intermediate_proxy_type(
    "Window",
    _Unusable,
    pd.core.window.rolling.Window,
)

Rolling = make_intermediate_proxy_type(
    "Rolling",
    cudf.core.window.Rolling,
    pd.core.window.Rolling,
)

ExponentialMovingWindow = make_intermediate_proxy_type(
    "ExponentialMovingWindow",
    cudf.core.window.ewm.ExponentialMovingWindow,
    pd.core.window.ewm.ExponentialMovingWindow,
)

ExponentialMovingWindowGroupby = make_intermediate_proxy_type(
    "ExponentialMovingWindowGroupby",
    _Unusable,
    pd.core.window.ewm.ExponentialMovingWindowGroupby,
)

EWMMeanState = make_intermediate_proxy_type(
    "EWMMeanState",
    _Unusable,
    pd.core.window.online.EWMMeanState,
)

Expanding = make_intermediate_proxy_type(
    "Expanding",
    _Unusable,
    pd.core.window.expanding.Expanding,
)

ExpandingGroupby = make_intermediate_proxy_type(
    "ExpandingGroupby",
    _Unusable,
    pd.core.window.expanding.ExpandingGroupby,
)

Resampler = make_intermediate_proxy_type(
    "Resampler", cudf.core.resample._Resampler, pd_Resampler
)

DataFrameResampler = make_intermediate_proxy_type(
    "DataFrameResampler", cudf.core.resample.DataFrameResampler, pd_Resampler
)

SeriesResampler = make_intermediate_proxy_type(
    "SeriesResampler", cudf.core.resample.SeriesResampler, pd_Resampler
)

StataReader = make_intermediate_proxy_type(
    "StataReader",
    _Unusable,
    pd.io.stata.StataReader,
)

HDFStore = make_final_proxy_type(
    "HDFStore",
    _Unusable,
    pd.HDFStore,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

ExcelFile = make_final_proxy_type(
    "ExcelFile",
    _Unusable,
    pd.ExcelFile,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

ExcelWriter = make_final_proxy_type(
    "ExcelWriter",
    _Unusable,
    pd.ExcelWriter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__fspath__": _FastSlowAttribute("__fspath__"),
    },
    bases=(os.PathLike,),
    metaclasses=(abc.ABCMeta,),
)

try:
    from pandas.io.formats.style import Styler as pd_Styler  # isort: skip

    Styler = make_final_proxy_type(
        "Styler",
        _Unusable,
        pd_Styler,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "css": _FastSlowAttribute("css"),
            "ctx": _FastSlowAttribute("ctx"),
            "index": _FastSlowAttribute("ctx"),
            "data": _FastSlowAttribute("data"),
            "_display_funcs": _FastSlowAttribute("_display_funcs"),
            "table_styles": _FastSlowAttribute("table_styles"),
        },
    )
except ImportError:
    # Styler requires Jinja to be installed
    pass

_eval_func = _FunctionProxy(_Unusable(), pd.eval)

register_proxy_func(pd.read_pickle)(
    _FunctionProxy(_Unusable(), pd.read_pickle)
)

register_proxy_func(pd.to_pickle)(_FunctionProxy(_Unusable(), pd.to_pickle))


def _get_eval_locals_and_globals(level, local_dict=None, global_dict=None):
    frame = sys._getframe(level + 3)
    local_dict = frame.f_locals if local_dict is None else local_dict
    global_dict = frame.f_globals if global_dict is None else global_dict
    return local_dict, global_dict


@register_proxy_func(pd.core.computation.eval.eval)
@nvtx.annotate(
    "CUDF_PANDAS_EVAL",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _eval(
    *args,
    parser="pandas",
    engine=None,
    local_dict=None,
    global_dict=None,
    **kwargs,
):
    # Custom implementation of to pre-process globals and
    # locals before calling pd.eval.
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _eval_func(
        *args,
        parser=parser,
        engine=engine,
        local_dict=local_dict,
        global_dict=global_dict,
        **kwargs,
    )


_orig_df_eval_method = DataFrame.eval


@register_proxy_func(pd.core.accessor.register_dataframe_accessor)
def _register_dataframe_accessor(name):
    return pd.core.accessor._register_accessor(name, DataFrame)


@register_proxy_func(pd.core.accessor.register_series_accessor)
def _register_series_accessor(name):
    return pd.core.accessor._register_accessor(name, Series)


@register_proxy_func(pd.core.accessor.register_index_accessor)
def _register_index_accessor(name):
    return pd.core.accessor._register_accessor(name, Index)


@nvtx.annotate(
    "CUDF_PANDAS_DATAFRAME_EVAL",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _df_eval_method(self, *args, local_dict=None, global_dict=None, **kwargs):
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _orig_df_eval_method(
        self, *args, local_dict=local_dict, global_dict=global_dict, **kwargs
    )


_orig_query_eval_method = DataFrame.query


@nvtx.annotate(
    "CUDF_PANDAS_DATAFRAME_QUERY",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _df_query_method(self, *args, local_dict=None, global_dict=None, **kwargs):
    # `query` API internally calls `eval`, hence we are making use of
    # helps of `eval` to populate locals and globals dict.
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _orig_query_eval_method(
        self, *args, local_dict=local_dict, global_dict=global_dict, **kwargs
    )


DataFrame.eval = _df_eval_method  # type: ignore
DataFrame.query = _df_query_method  # type: ignore

_JsonReader = make_intermediate_proxy_type(
    "_JsonReader",
    _Unusable,
    pd.io.json._json.JsonReader,
)

_TextFileReader = make_intermediate_proxy_type(
    "_TextFileReader", _Unusable, pd.io.parsers.readers.TextFileReader
)

_XportReader = make_intermediate_proxy_type(
    "_XportReader", _Unusable, pd_XportReader
)

_SAS7BDATReader = make_intermediate_proxy_type(
    "_SAS7BDATReader", _Unusable, pd_SAS7BDATReader
)

USFederalHolidayCalendar = make_final_proxy_type(
    "USFederalHolidayCalendar",
    _Unusable,
    pd_USFederalHolidayCalendar,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

HolidayCalendarMetaClass = make_final_proxy_type(
    "HolidayCalendarMetaClass",
    _Unusable,
    pd_HolidayCalendarMetaClass,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)


@register_proxy_func(pd_HolidayCalendarFactory)
def holiday_calendar_factory_wrapper(*args, **kwargs):
    # Call the original HolidayCalendarFactory
    result = _FunctionProxy(_Unusable(), pd_HolidayCalendarFactory)(
        *args, **kwargs
    )
    # Return the slow proxy of the result
    return result._fsproxy_slow


AbstractHolidayCalendar = make_final_proxy_type(
    "AbstractHolidayCalendar",
    _Unusable,
    pd_AbstractHolidayCalendar,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
    metaclasses=(pd_HolidayCalendarMetaClass,),
)

Holiday = make_final_proxy_type(
    "Holiday",
    _Unusable,
    pd_Holiday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)
USThanksgivingDay = make_final_proxy_type(
    "USThanksgivingDay",
    _Unusable,
    pd_USThanksgivingDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

USColumbusDay = make_final_proxy_type(
    "USColumbusDay",
    _Unusable,
    pd_USColumbusDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

USLaborDay = make_final_proxy_type(
    "USLaborDay",
    _Unusable,
    pd_USLaborDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

USMemorialDay = make_final_proxy_type(
    "USMemorialDay",
    _Unusable,
    pd_USMemorialDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

USMartinLutherKingJr = make_final_proxy_type(
    "USMartinLutherKingJr",
    _Unusable,
    pd_USMartinLutherKingJr,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

USPresidentsDay = make_final_proxy_type(
    "USPresidentsDay",
    _Unusable,
    pd_USPresidentsDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)


GoodFriday = make_final_proxy_type(
    "GoodFriday",
    _Unusable,
    pd_GoodFriday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

EasterMonday = make_final_proxy_type(
    "EasterMonday",
    _Unusable,
    pd_EasterMonday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

FY5253 = make_final_proxy_type(
    "FY5253",
    _Unusable,
    pd.offsets.FY5253,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BDay = make_final_proxy_type(
    "BDay",
    _Unusable,
    pd.offsets.BDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BMonthBegin = make_final_proxy_type(
    "BMonthBegin",
    _Unusable,
    pd.offsets.BMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BMonthEnd = make_final_proxy_type(
    "BMonthEnd",
    _Unusable,
    pd.offsets.BMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BQuarterBegin = make_final_proxy_type(
    "BQuarterBegin",
    _Unusable,
    pd.offsets.BQuarterBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BQuarterEnd = make_final_proxy_type(
    "BQuarterEnd",
    _Unusable,
    pd.offsets.BQuarterEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BusinessDay = make_final_proxy_type(
    "BusinessDay",
    _Unusable,
    pd.offsets.BusinessDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BusinessHour = make_final_proxy_type(
    "BusinessHour",
    _Unusable,
    pd.offsets.BusinessHour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BusinessMonthBegin = make_final_proxy_type(
    "BusinessMonthBegin",
    _Unusable,
    pd.offsets.BusinessMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BusinessMonthEnd = make_final_proxy_type(
    "BusinessMonthEnd",
    _Unusable,
    pd.offsets.BusinessMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BYearBegin = make_final_proxy_type(
    "BYearBegin",
    _Unusable,
    pd.offsets.BYearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BYearEnd = make_final_proxy_type(
    "BYearEnd",
    _Unusable,
    pd.offsets.BYearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CBMonthBegin = make_final_proxy_type(
    "CBMonthBegin",
    _Unusable,
    pd.offsets.CBMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CBMonthEnd = make_final_proxy_type(
    "CBMonthEnd",
    _Unusable,
    pd.offsets.CBMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CDay = make_final_proxy_type(
    "CDay",
    _Unusable,
    pd.offsets.CDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CustomBusinessDay = make_final_proxy_type(
    "CustomBusinessDay",
    _Unusable,
    pd.offsets.CustomBusinessDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CustomBusinessHour = make_final_proxy_type(
    "CustomBusinessHour",
    _Unusable,
    pd.offsets.CustomBusinessHour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CustomBusinessMonthBegin = make_final_proxy_type(
    "CustomBusinessMonthBegin",
    _Unusable,
    pd.offsets.CustomBusinessMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

CustomBusinessMonthEnd = make_final_proxy_type(
    "CustomBusinessMonthEnd",
    _Unusable,
    pd.offsets.CustomBusinessMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

DateOffset = make_final_proxy_type(
    "DateOffset",
    _Unusable,
    pd.offsets.DateOffset,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

BaseOffset = make_final_proxy_type(
    "BaseOffset",
    _Unusable,
    pd.offsets.BaseOffset,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Day = make_final_proxy_type(
    "Day",
    _Unusable,
    pd.offsets.Day,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Easter = make_final_proxy_type(
    "Easter",
    _Unusable,
    pd.offsets.Easter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

FY5253Quarter = make_final_proxy_type(
    "FY5253Quarter",
    _Unusable,
    pd.offsets.FY5253Quarter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Hour = make_final_proxy_type(
    "Hour",
    _Unusable,
    pd.offsets.Hour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

LastWeekOfMonth = make_final_proxy_type(
    "LastWeekOfMonth",
    _Unusable,
    pd.offsets.LastWeekOfMonth,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Micro = make_final_proxy_type(
    "Micro",
    _Unusable,
    pd.offsets.Micro,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Milli = make_final_proxy_type(
    "Milli",
    _Unusable,
    pd.offsets.Milli,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Minute = make_final_proxy_type(
    "Minute",
    _Unusable,
    pd.offsets.Minute,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)


MonthBegin = make_final_proxy_type(
    "MonthBegin",
    _Unusable,
    pd.offsets.MonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

MonthEnd = make_final_proxy_type(
    "MonthEnd",
    _Unusable,
    pd.offsets.MonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Nano = make_final_proxy_type(
    "Nano",
    _Unusable,
    pd.offsets.Nano,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

QuarterBegin = make_final_proxy_type(
    "QuarterBegin",
    _Unusable,
    pd.offsets.QuarterBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

QuarterEnd = make_final_proxy_type(
    "QuarterEnd",
    _Unusable,
    pd.offsets.QuarterEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Second = make_final_proxy_type(
    "Second",
    _Unusable,
    pd.offsets.Second,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

SemiMonthBegin = make_final_proxy_type(
    "SemiMonthBegin",
    _Unusable,
    pd.offsets.SemiMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

SemiMonthEnd = make_final_proxy_type(
    "SemiMonthEnd",
    _Unusable,
    pd.offsets.SemiMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Tick = make_final_proxy_type(
    "Tick",
    _Unusable,
    pd.offsets.Tick,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Week = make_final_proxy_type(
    "Week",
    _Unusable,
    pd.offsets.Week,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

WeekOfMonth = make_final_proxy_type(
    "WeekOfMonth",
    _Unusable,
    pd.offsets.WeekOfMonth,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

YearBegin = make_final_proxy_type(
    "YearBegin",
    _Unusable,
    pd.offsets.YearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

YearEnd = make_final_proxy_type(
    "YearEnd",
    _Unusable,
    pd.offsets.YearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

Flags = make_final_proxy_type(
    "Flags",
    _Unusable,
    pd.Flags,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

NamedAgg = make_final_proxy_type(
    "NamedAgg",
    _Unusable,
    pd.NamedAgg,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

ArrowExtensionArray = make_final_proxy_type(
    "ExtensionArray",
    _Unusable,
    pd.arrays.ArrowExtensionArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)


# The following are subclasses of `pandas.core.base.PandasObj`,
# excluding subclasses defined in `pandas.core.internals`.  These are
# not strictly part of the Pandas public API, but they do appear as
# return types.

_PANDAS_OBJ_FINAL_TYPES = [
    pd.core.arrays.sparse.array.SparseArray,
    pd.core.indexes.frozen.FrozenList,
    pd.core.indexes.category.CategoricalIndex,
    pd.core.indexes.datetimelike.DatetimeTimedeltaMixin,
    pd.core.indexes.datetimelike.DatetimeIndexOpsMixin,
    pd.core.indexes.extension.NDArrayBackedExtensionIndex,
    pd.core.generic.NDFrame,
    pd.core.indexes.accessors.PeriodProperties,
    pd.core.indexes.accessors.Properties,
    pd.plotting._core.PlotAccessor,
    pd.io.sql.SQLiteTable,
    pd.io.sql.SQLTable,
    pd.io.sql.SQLDatabase,
    pd.io.sql.SQLiteDatabase,
    pd.io.sql.PandasSQL,
]

_PANDAS_OBJ_INTERMEDIATE_TYPES = [
    pd.core.groupby.groupby.GroupByPlot,
    pd.core.groupby.groupby.GroupBy,
    pd.core.groupby.groupby.BaseGroupBy,
]

for typ in _PANDAS_OBJ_FINAL_TYPES:
    if typ.__name__ in globals():
        # if we already defined a proxy type
        # corresponding to this type, use that.
        continue
    globals()[typ.__name__] = make_final_proxy_type(
        typ.__name__,
        _Unusable,
        typ,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__array__": array_method,
            "__array_function__": array_function_method,
            "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
            "__hash__": _FastSlowAttribute("__hash__"),
        },
    )


for typ in _PANDAS_OBJ_INTERMEDIATE_TYPES:
    if typ.__name__ in globals():
        # if we already defined a proxy type
        # corresponding to this type, use that.
        continue
    globals()[typ.__name__] = make_intermediate_proxy_type(
        typ.__name__,
        _Unusable,
        typ,
    )


# timestamps and timedeltas are not proxied, but non-proxied
# pandas types are currently not picklable. Thus, we define
# custom reducer/unpicker functions for these types:
def _reduce_obj(obj):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        # args can contain objects that are unpicklable
        # when the module accelerator is disabled
        # (freq is of a proxy type):
        pickled_args = pickle.dumps(obj.__reduce__())

    return _unpickle_obj, (pickled_args,)


def _unpickle_obj(pickled_args):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        unpickler, args = pickle.loads(pickled_args)
    obj = unpickler(*args)
    return obj


copyreg.dispatch_table[pd.Timestamp] = _reduce_obj
# same reducer/unpickler can be used for Timedelta:
copyreg.dispatch_table[pd.Timedelta] = _reduce_obj
