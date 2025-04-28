# Copyright (c) 2022-2025, NVIDIA CORPORATION.

import inspect


# `functools.partialmethod` does not allow setting attributes such as
# __doc__ on the resulting method. So we use a simple alternative to
# it here:
def _partialmethod(method, *args1, **kwargs1):
    def wrapper(self, *args2, **kwargs2):
        return method(self, *args1, *args2, **kwargs1, **kwargs2)

    return wrapper


class Operation:
    """Descriptor used to define operations for delegating mixins.

    This class is designed to be assigned to the attributes (the delegating
    methods) defined by the OperationMixin. This class will create the method
    and mimic all the expected attributes for that method to appear as though
    it was originally designed on the class. The use of the descriptor pattern
    ensures that the method is only created the first time it is invoked, after
    which all further calls use the callable generated on the first invocation.

    Parameters
    ----------
    name : str
        The name of the operation.
    docstring_format_args : str
        The attribute of the owning class from which to pull format parameters
        for this operation's docstring.
    base_operation : str
        The underlying operation function to be invoked when operation `name`
        is called on the owning class.
    """

    def __init__(self, name, docstring_format_args, base_operation):
        self._name = name
        self._docstring_format_args = docstring_format_args
        self._base_operation = base_operation

    def __get__(self, obj, owner=None):
        retfunc = _partialmethod(self._base_operation, op=self._name)

        # Required attributes that will exist.
        retfunc.__name__ = self._name
        retfunc.__qualname__ = ".".join([owner.__name__, self._name])
        retfunc.__module__ = self._base_operation.__module__

        if self._base_operation.__doc__ is not None:
            retfunc.__doc__ = self._base_operation.__doc__.format(
                cls=owner.__name__,
                op=self._name,
                **self._docstring_format_args,
            )

        retfunc.__annotations__ = self._base_operation.__annotations__.copy()
        retfunc.__annotations__.pop("op", None)
        retfunc_params = [
            v
            for k, v in inspect.signature(
                self._base_operation
            ).parameters.items()
            if k != "op"
        ]
        retfunc.__signature__ = inspect.Signature(retfunc_params)

        setattr(owner, self._name, retfunc)

        if obj is None:
            return getattr(owner, self._name)
        else:
            return getattr(obj, self._name)


def _should_define_operation(cls, operation, base_operation_name):
    if operation not in dir(cls):
        return True

    # If the class doesn't override the base operation we stick to whatever
    # parent implementation exists.
    if base_operation_name not in cls.__dict__:
        return False

    # At this point we know that the class has the operation defined but it
    # also overrides the base operation. Since this function is called before
    # the operation is defined on the current class, we know that it inherited
    # the operation from a parent. We therefore have three possibilities:
    # 1. A parent class manually defined the operation. That override takes
    #    precedence even if the current class defined the base operation.
    # 2. A parent class has an auto-generated operation, i.e. it is of type
    #    Operation and was created by OperationMixin.__init_subclass__. The
    #    current class must override it so that its base operation is used
    #    rather than the parent's base operation.
    # 3. The method is defined for all classes, i.e. it is a method of object.
    for base_cls in cls.__mro__:
        # We always override methods defined for object.
        if base_cls is object:
            return True
        # The first attribute in the MRO is the one that will be used.
        if operation in base_cls.__dict__:
            return isinstance(base_cls.__dict__[operation], Operation)

    # This line should be unreachable since we know the attribute exists
    # somewhere in the MRO if the for loop was entered.
    assert False, "Operation attribute not found in hierarchy."


def _create_delegating_mixin(
    mixin_name,
    docstring,
    category_name,
    base_operation_name,
    supported_operations,
):
    """Factory for mixins defining collections of delegated operations.

    This function generates mixins based on two common paradigms in cuDF:

    1. libcudf groups many operations into categories using a common API. These
       APIs usually accept an enum to delineate the specific operation to
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
        name will be used to define or access the following attributes as shown
        in the example below:
            - f'_{category_name}_DOCSTRINGS'
            - f'_VALID_{category_name}S'  # The subset of ops a subclass allows
            - f'_SUPPORTED_{category_name}S'  # The ops supported by the mixin
    base_operation_name : str
        The name given to the core function implementing this category of
        operations.  The corresponding function is the entrypoint for child
        classes.
    supported_ops : List[str]
        The list of valid operations that subclasses of the resulting mixin may
        request to be implemented.

    Examples
    --------
    >>> # The class below:
    >>> class Person:
    ...     def _greet(self, op):
    ...         print(op)
    ...
    ...     def hello(self):
    ...         self._greet("hello")
    ...
    ...     def goodbye(self):
    ...         self._greet("goodbye")
    >>> # can  be rewritten using a delegating mixin as follows:
    >>> Greeter = _create_delegating_mixin(
    ...     "Greeter", "", "GREETING", "_greet", {"hello", "goodbye", "hey"}
    ... )
    >>> # The `hello` and `goodbye` methods will now be automatically generated
    >>> # for the Person class below.
    >>> class Person(Greeter):
    ...     _VALID_GREETINGS = {"hello", "goodbye"}
    ...
    ...     def _greet(self, op: str):
    ...         '''Say {op}.'''
    ...         print(op)
    >>> mom = Person()
    >>> mom.hello()
    hello
    >>> # The Greeter class could also enable the `hey` method, but Person did
    >>> # not include it in the _VALID_GREETINGS set so it will not exist.
    >>> mom.hey()
    Traceback (most recent call last):
        ...
    AttributeError: 'Person' object has no attribute 'hey'
    >>> # The docstrings for each method are generated by formatting the _greet
    >>> # docstring with the operation name as well as any additional keys
    >>> # provided via the _GREETING_DOCSTRINGS parameter.
    >>> print(mom.hello.__doc__)
    Say hello.
    """
    # The first two attributes may be defined on subclasses of the generated
    # OperationMixin to indicate valid attributes and parameters to use when
    # formatting docstrings. The supported_attr will be defined on the
    # OperationMixin itself to indicate what operations its subclass may
    # inherit from it.
    validity_attr = f"_VALID_{category_name}S"
    docstring_attr = f"_{category_name}_DOCSTRINGS"
    supported_attr = f"_SUPPORTED_{category_name}S"

    class OperationMixin:
        @classmethod
        def __init_subclass__(cls):
            # Support composition of various OperationMixins. Note that since
            # this __init_subclass__ is defined on mixins, it does not prohibit
            # classes that inherit it from implementing this method as well as
            # long as those implementations also include this super call.
            super().__init_subclass__()

            # Only add the valid set of operations for a particular class.
            valid_operations = set()
            for base_cls in cls.__mro__:
                # Check for sentinel indicating that all operations are valid.
                valid_operations |= getattr(base_cls, validity_attr, set())

            invalid_operations = valid_operations - supported_operations
            assert len(invalid_operations) == 0, (
                f"Invalid requested operations: {invalid_operations}"
            )

            base_operation = getattr(cls, base_operation_name)
            for operation in valid_operations:
                if _should_define_operation(
                    cls, operation, base_operation_name
                ):
                    docstring_format_args = getattr(
                        cls, docstring_attr, {}
                    ).get(operation, {})
                    op_attr = Operation(
                        operation, docstring_format_args, base_operation
                    )
                    setattr(cls, operation, op_attr)

    OperationMixin.__name__ = mixin_name
    OperationMixin.__qualname__ = mixin_name
    OperationMixin.__doc__ = docstring

    def _operation(self, op: str, *args, **kwargs):
        raise NotImplementedError

    _operation.__name__ = base_operation_name
    _operation.__qualname__ = ".".join([mixin_name, base_operation_name])
    _operation.__doc__ = (
        f"The core {category_name.lower()} function. Must be overridden by "
        "subclasses, the default implementation raises a NotImplementedError."
    )

    setattr(OperationMixin, base_operation_name, _operation)
    # Making this attribute available makes it easy for subclasses to indicate
    # that all supported operations for this mixin are valid.
    setattr(OperationMixin, supported_attr, supported_operations)

    return OperationMixin
