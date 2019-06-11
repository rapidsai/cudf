from cudf.bindings.rolling import apply_rolling

class Rolling:

    def __init__(self, obj, window):
        self.obj = obj
        self.window = window

    def sum(self):
        result_col = apply_rolling(
            self.obj._column,
            self.window,
            self.window,
            "sum")
        return self.obj._copy_construct(data=result_col)
