from .series_impl import SeriesImpl


class NumericalSeriesImpl(SeriesImpl):
    def __init__(self, dtype):
        self._dtype = dtype

    def element_to_str(self, value):
        return str(value)

