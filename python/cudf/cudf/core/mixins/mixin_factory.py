# Copyright (c) 2022, NVIDIA CORPORATION.

import inspect


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

    class OperationMixin:
        @classmethod
        def _add_operation(cls, operation):
            # This function creates operations on-the-fly.

            # Generate a signature without the `op` parameter.
            base_operation = getattr(cls, operation_name)
            base_params = inspect.signature(base_operation).parameters.values()
            params = [p for p in base_params if p.name != "op"]
            signature = inspect.Signature(parameters=params)

            # Generate the list of arguments forwarded to the base operation.
            arglist = ", ".join(
                [
                    f"{key}={key}"
                    for key in signature.parameters
                    if key not in ("self", "args", "kwargs")
                ]
            )
            arglist = arglist + (", " if arglist else "") + "*args, **kwargs"

            # Apply the formatted docstring of the base operation to the
            # operation being created here.
            docstring = base_operation.__doc__.format(
                cls=cls.__name__,
                op=operation,
                **getattr(cls, docstring_attr, {}).get(operation, {}),
            )

            namespace = {}
            exec_str = f"""
def {operation}{str(signature)}:
    '''{docstring}
    '''
    return self.{operation_name}(op="{operation}", {arglist})
"""
            exec(exec_str, namespace)
            setattr(cls, operation, namespace[operation])

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
                # Check if the operation is already defined so that subclasses
                # can override the method if desired.
                if not hasattr(cls, operation):
                    cls._add_operation(operation)

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
