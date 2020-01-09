from cudf._libxx.table import _Table


class Table(_Table):
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
