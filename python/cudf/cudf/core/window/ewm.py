from pandas.core.window.ewm import get_center_of_mass

from cudf._lib.reduce import scan
from cudf.core.window.rolling import _RollingBase


class ExponentialMovingWindow(_RollingBase):
    def __init__(
        self,
        obj,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        axis=0,
    ):
        self.obj = obj
        self.adjust = adjust
        self.com = get_center_of_mass(com, span, halflife, alpha)

    def mean(self):
        return self._apply_agg("ewma")

    def var(self, bias):
        self.bias = bias
        return self._apply_agg("ewmvar")

    def std(self, bias):
        self.bias = bias
        return self._apply_agg("ewmstd")

    def corr(self, other):
        raise NotImplementedError("corr not yet supported.")

    def cov(self, other):
        raise NotImplementedError("cov not yet supported.")

    def _apply_agg_series(self, sr, agg_name):
        return scan(agg_name, sr._column, True, com=self.com, adjust=self.adjust)
