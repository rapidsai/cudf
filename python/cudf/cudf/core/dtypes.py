from pandas.api.extensions import ExtensionDtype


class CategoricalDtype(ExtensionDtype):
    def __init__(self, categories=None, ordered=False):
        self.categories = categories
        self.ordered = ordered

    @property
    def type(self):
        return self.categories.dtype.type

    @property
    def name(self):
        return "category"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, CategoricalDtype):
            return False
        return self.categories.equals(other.categories) and (
            self.ordered == other.ordered
        )
