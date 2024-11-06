# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.types import (
    DataType,
    Interpolation,
    NanEquality,
    NullEquality,
    NullOrder,
    NullPolicy,
    Order,
)

class Kind(IntEnum):
    SUM = auto()
    PRODUCT = auto()
    MIN = auto()
    MAX = auto()
    COUNT_VALID = auto()
    COUNT_ALL = auto()
    ANY = auto()
    ALL = auto()
    SUM_OF_SQUARES = auto()
    MEAN = auto()
    VARIANCE = auto()
    STD = auto()
    MEDIAN = auto()
    QUANTILE = auto()
    ARGMAX = auto()
    ARGMIN = auto()
    NUNIQUE = auto()
    NTH_ELEMENT = auto()
    RANK = auto()
    COLLECT_LIST = auto()
    COLLECT_SET = auto()
    PTX = auto()
    CUDA = auto()
    CORRELATION = auto()
    COVARIANCE = auto()

class CorrelationType(IntEnum):
    PEARSON = auto()
    KENDALL = auto()
    SPEARMAN = auto()

class EWMHistory(IntEnum):
    INFINITE = auto()
    FINITE = auto()

class RankMethod(IntEnum):
    FIRST = auto()
    AVERAGE = auto()
    MIN = auto()
    MAX = auto()
    DENSE = auto()

class RankPercentage(IntEnum):
    NONE = auto()
    ZERO_NORMALIZED = auto()
    ONE_NORMALIZED = auto()

class UdfType(IntEnum):
    CUDA = auto()
    PTX = auto()

class Aggregation:
    def kind(self) -> Kind: ...

def sum() -> Aggregation: ...
def product() -> Aggregation: ...
def min() -> Aggregation: ...
def max() -> Aggregation: ...
def count(null_handling: NullPolicy = NullPolicy.INCLUDE) -> Aggregation: ...
def any() -> Aggregation: ...
def all() -> Aggregation: ...
def sum_of_squares() -> Aggregation: ...
def mean() -> Aggregation: ...
def variance(ddof: int = 1) -> Aggregation: ...
def std(ddof: int = 1) -> Aggregation: ...
def median() -> Aggregation: ...
def quantile(
    quantiles: list[float], interp: Interpolation = Interpolation.LINEAR
) -> Aggregation: ...
def argmax() -> Aggregation: ...
def argmin() -> Aggregation: ...
def ewma(center_of_mass: float, history: EWMHistory) -> Aggregation: ...
def nunique(null_handling: NullPolicy = NullPolicy.EXCLUDE) -> Aggregation: ...
def nth_element(
    n: int, null_handling: NullPolicy = NullPolicy.INCLUDE
) -> Aggregation: ...
def collect_list(
    null_handling: NullPolicy = NullPolicy.INCLUDE,
) -> Aggregation: ...
def collect_set(
    null_handling: NullPolicy = NullPolicy.INCLUDE,
    nulls_equal: NullEquality = NullEquality.EQUAL,
    nans_equal: NanEquality = NanEquality.ALL_EQUAL,
) -> Aggregation: ...
def udf(operation: str, output_type: DataType) -> Aggregation: ...
def correlation(type: CorrelationType, min_periods: int) -> Aggregation: ...
def covariance(min_periods: int, ddof: int) -> Aggregation: ...
def rank(
    method: RankMethod,
    column_order: Order = Order.ASCENDING,
    null_handling: NullPolicy = NullPolicy.EXCLUDE,
    null_precedence: NullOrder = NullOrder.AFTER,
    percentage: RankPercentage = RankPercentage.NONE,
) -> Aggregation: ...
