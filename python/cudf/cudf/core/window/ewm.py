from pandas.core.window.ewm import get_center_of_mass

from cudf._lib.reduce import scan
from cudf.core.window.rolling import _RollingBase
from cudf.api.types import is_numeric_dtype

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
        self.bias=None

    def mean(self):
        return self._apply_agg("ewma")

    def var(self, bias):
        raise NotImplementedError("ewmvar not yet supported.")

    def std(self, bias):
        raise NotImplementedError("ewmstd not yet supported.")

    def corr(self, other):
        raise NotImplementedError("corr not yet supported.")

    def cov(self, other):
        raise NotImplementedError("cov not yet supported.")

    def _apply_agg_series(self, sr, agg_name):

        if not is_numeric_dtype(sr.dtype):
            raise TypeError("No numeric types to aggregate")

        kws = {
            "com": self.com,
            "adjust": self.adjust,
        }
        if self.bias is not None:
            kws['bias'] = self.bias

        return scan(agg_name, sr._column.astype('float64'), True, **kws)
