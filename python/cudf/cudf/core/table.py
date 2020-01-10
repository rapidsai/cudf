from collections import OrderedDict

from cudf._libxx.table import _Table


class OrderedColumnDict(OrderedDict):
    def __setitem__(self, key, value):
        from cudf.core.column import ColumnBase

        if not isinstance(value, ColumnBase):
            raise TypeError(
                f"Cannot insert object of type "
                f"{value.__class__.__name__} into OrderedColumnDict"
            )

        if self.first is not None and len(self.first) > 0:
            if len(value) != len(self.first):
                raise ValueError(
                    f"Cannot insert Column of different length "
                    "into OrderedColumnDict"
                )

        super().__setitem__(key, value)

    @property
    def first(self):
        """
        Returns the first value if self is non-empty;
        returns None otherwise.
        """
        if len(self) == 0:
            return None
        else:
            return next(iter(self.values()))


class NamedTable(_Table):
    def __init__(self, data=None):
        """
        Data: an OrderedColumnDict of columns
        """
        if data is None:
            data = OrderedColumnDict()
        self._data = data
        super().__init__(self._data.values())

    def _unaryop(self, op):
        result = self.copy()
        for name, col in result._data.items():
            result._data[name] = col.unary_operator(op)
        return result

    def sin(self):
        return self._unaryop("sin")

    def cos(self):
        return self._unaryop("cos")

    def tan(self):
        return self._unaryop("tan")

    def asin(self):
        return self._unaryop("asin")

    def acos(self):
        return self._unaryop("acos")

    def atan(self):
        return self._unaryop("atan")

    def exp(self):
        return self._unaryop("exp")

    def log(self):
        return self._unaryop("log")

    def sqrt(self):
        return self._unaryop("sqrt")
