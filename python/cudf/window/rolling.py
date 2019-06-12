import cudf
from cudf.bindings.rolling import apply_rolling

class Rolling:

    def __init__(self, obj, window):
        self.obj = obj
        self.window = window
        self._validate_args()

    def _apply_agg_series(self, sr, agg_name):
        result_col = apply_rolling(
            sr._column,
            self.window,
            self.window,
            agg_name)
        return sr._copy_construct(data=result_col)

    def _apply_agg_dataframe(self, df, agg_name):
        result_df = cudf.DataFrame({})
        for col_name in df.columns:
            result_df.add_column(name=col_name,
                                 data=self._apply_agg_series(df[col_name], agg_name))
        result_df.index = df.index
        return result_df

    def _apply_agg(self, agg_name):
        if isinstance(self.obj, cudf.Series):
            return self._apply_agg_series(self.obj, agg_name)
        else:
            return self._apply_agg_dataframe(self.obj, agg_name)

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

    def _validate_args(self):

        if self.window <= 0:
            raise ValueError("Window size cannot be zero or negative")
