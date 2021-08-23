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

    def var(self):
        return self._apply_agg("var")

    def std(self):
        return self._apply_agg("std")

    def corr(self):
        return self._apply_agg("corr")

    def cov(self):
        return self._apply_agg("cov")

    def _apply_agg_series(self, sr, agg_name):
        result = scan('ewma', sr._column, True, com=self.com, adjust=self.adjust)
        return result
