from pandas.api.extensions import ExtensionDtype, register_extension_dtype


@register_extension_dtype
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

    @property
    def str(self):
        return "|O08"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """
        Rules for equality largely borrowed from pd.CategoricalDtype
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        elif self.categories is None or other.categories is None:
            return True
        elif self.ordered or other.ordered:
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            if (
                self.categories.dtype == other.categories.dtype
                and self.categories.equals(other.categories)
            ):
                return True
        return False

    def construct_from_string():
        pass
