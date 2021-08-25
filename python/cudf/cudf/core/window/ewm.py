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
        return self._apply_agg("mean")

    def var(self, bias):
        self.bias = bias
        return self._apply_agg("var")

    def std(self, bias):
        self.bias = bias
        return self._apply_agg("std")

    def corr(self, other):
        raise NotImplementedError("corr not yet supported.")

    def cov(self, other):
        raise NotImplementedError("cov not yet supported.")

    def _apply_agg_series(self, sr, agg_name):
        if agg_name == 'mean':
            result = scan('ewma', sr._column, True, com=self.com, adjust=self.adjust)
        elif agg_name == 'var':
            result = scan("ewmvar", sr._column, True, com=self.com, adjust=self.adjust)
        else:
            result = None
        return result
