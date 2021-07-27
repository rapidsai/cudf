from pandas.core.window.ewm import get_center_of_mass

class ExponentialMovingWindow(object):
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
        axis=0
    ):

        self._obj = obj
        self.com = get_center_of_mass(com, span, halflife, alpha)
