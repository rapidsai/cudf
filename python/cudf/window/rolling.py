from cudf.bindings.rolling import apply_rolling

class Rolling:

    def __init__(self, obj, window):
        self.obj = obj
        self.window = window

    def _apply_agg(self, agg_name):
        result_col = apply_rolling(
            self.obj._column,
            self.window,
            self.window,
            agg_name)
        return self.obj._copy_construct(data=result_col)

    def sum(self):
        return self._apply_agg("sum")

    def min(self):
        return self._apply_agg("min")

    def max(self):
        return self._apply_agg("max")

    def mean(self):
        return self._apply_agg("mean")

    def count(self):
        return self._apply_agg("count")
