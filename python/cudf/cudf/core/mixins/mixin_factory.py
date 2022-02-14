# Copyright (c) 2022, NVIDIA CORPORATION.

import functools
import inspect


# A simple `partialmethod` that allows setting attributes such as
# __doc__ on instances.
def _partialmethod(method, *args1, **kwargs1):
    @functools.wraps(method)
    def wrapper(self, *args2, **kwargs2):
        return method(self, *args1, *args2, **kwargs1, **kwargs2)

    return wrapper


def _create_delegating_mixin(
    mixin_name, docstring, category_name, operation_name, supported_operations
):
    """Factory for mixins defining collections of delegated operations.

    This function generates mixins based on two common paradigms in cuDF:

    1. libcudf groups many operations into categories using a common API. These
       APIs usually accept an enum to delinate the specific operation to
       perform, e.g. binary operations use the `binary_operator` enum when
       calling the `binary_operation` function. cuDF Python mimics this
       structure by having operations within a category delegate to a common
       internal function (e.g. DataFrame.__add__ calls DataFrame._binaryop).
    2. Many cuDF classes implement similar operations (e.g. `sum`) via
       delegation to lower-level APIs before reaching a libcudf C++ function
       call. As a result, many API function calls actually involve multiple
       delegations to lower-level APIs that can look essentially identical. An
       example of such a sequence would be DataFrame.sum -> DataFrame._reduce
       -> Column.sum -> Column._reduce -> libcudf.

    This factory creates mixins for a category of operations implemented by via
    this delegator pattern. The resulting mixins make it easy to share common
    functions across various classes while also providing a common entrypoint
    for implementing the centralized logic for a given category of operations.
    Its usage is best demonstrated by example below.

    Parameters
    ----------
    mixin_name : str
        The name of the class. This argument should be the same as the object
        that this function's output is assigned to, e.g.
        :code:`Baz = _create_delegating_mixin("Baz", ...)`.
    docstring : str
        The documentation string for the mixin class.
    category_name : str
        The category of operations for which a mixin is being created. This
        name will be used to define the following attributes as shown in the
        example below:
            - f'_{category_name}_DOCSTRINGS'
            - f'_VALID_{category_name}S'
            - f'_SUPPORTED_{category_name}S'
    operation_name : str
        The name given to the core function implementing this category of
        operations.  The corresponding function is the entrypoint for child
        classes.
    supported_ops : List[str]
        The list of valid operations that subclasses of the resulting mixin may
        request to be implemented.

    Examples
    --------
    >>> MyFoo = _create_delegating_mixin(
    ...     "MyFoo", "MyFoo's docstring", "FOO", "do_foo", {"foo1", "foo2"}
    ... )

    >>> # The above is equivalent to defining a class like so:
    ... class MyFoo:
    ...     # The set of valid foo operations.
    ...     _VALID_FOOS = {"foo1", "foo2"}
    ...
    ...     # This is the method for child classes to override. Note that the
    ...     # first parameter is always called "op".
    ...     def do_foo(self, op, *args, **kwargs):
    ...         raise NotImplementedError

    >>> # MyFoo can be used as follows.
    >>> class BarImplementsFoo(MyFoo):
    ...     _SUPPORTED_FOOS = ("foo1",)
    ...     _FOO_DOCSTRINGS = {"ret": "42"}
    ...
    ...     # This method's docstring will be formatted and used for all valid
    ...     # operations. The `op` key is always automatically filled in, while
    ...     # other keys are formatted using _FOO_DOCSTRINGS.
    ...     def do_foo(self, op, *args, **kwargs):
    ...         '''Perform the operation {op}, which returns {ret}.'''
    ...         return 42

    >>> bar = BarImplementsFoo()
    >>> print(bar.foo1())
    42

    >>> # This will raise an AttributeError because foo2 is not supported by
    >>> # the BarImplementsFoo subclass of MyFoo.
    >>> # print(bar.foo2())

    >>> # The docstring is formatted with the operation name as well as any
    >>> # additional keys provided via the _FOO_DOCSTRINGS parameter.
    >>> print(bar.foo1.__doc__)
    Perform the operation foo1, which returns 42.
    """
    docstring_attr = f"{category_name}_DOCSTRINGS"
    validity_attr = f"_VALID_{category_name}S"

    class Operation:
        """
        """

        def __init__(self):
            self._operation_name = operation_name

        def __set_name__(self, owner, name):
            self._name = name

        def __get__(self, obj, owner=None):
            base_operation = getattr(owner, self._operation_name)
            retfunc = _partialmethod(base_operation, op=self._name)

            retfunc.__doc__ = base_operation.__doc__.format(
                cls=owner.__name__,
                op=self._name,
                **getattr(owner, docstring_attr, {}).get(self._name, {}),
            )
            retfunc.__name__ = self._name
            retfunc_params = {
                k: v
                for k, v in inspect.signature(
                    base_operation
                ).parameters.items()
                if k != "op"
            }.values()
            retfunc.__signature__ = inspect.Signature(retfunc_params)

            setattr(owner, self._name, retfunc)

            if obj is None:
                return getattr(owner, self._name)
            else:
                return getattr(obj, self._name)

    class OperationMixin:
        @classmethod
        def __init_subclass__(cls):
            # Only add the valid set of operations for a particular class.
            valid_operations = set()
            for base_cls in cls.__mro__:
                valid_operations |= getattr(base_cls, validity_attr, set())

            invalid_operations = valid_operations - supported_operations
            assert (
                len(invalid_operations) == 0
            ), f"Invalid requested operations: {invalid_operations}"

            for operation in valid_operations:
                if operation not in dir(cls):
                    op_attr = Operation()
                    setattr(cls, operation, op_attr)
                    op_attr.__set_name__(cls, operation)

    def _operation(self, op: str, *args, **kwargs):
        raise NotImplementedError

    OperationMixin.__name__ = mixin_name
    OperationMixin.__doc__ = docstring
    setattr(OperationMixin, operation_name, _operation)
    # This attribute is set in case lookup is convenient at a later point, but
    # it is not strictly necessary since `supported_operations` is part of the
    # closure associated with the class's creation.
    setattr(
        OperationMixin, f"_SUPPORTED_{category_name}S", supported_operations
    )

    return OperationMixin
