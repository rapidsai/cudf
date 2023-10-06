# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import functools
import inspect
import operator
import pickle
import types
from collections.abc import Iterator
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
)

from .annotation import nvtx

_XDF_NVTX_COLORS = {
    "COPY_SLOW_TO_FAST": 0xCA0020,
    "COPY_FAST_TO_SLOW": 0xF4A582,
    "EXECUTE_FAST": 0x92C5DE,
    "EXECUTE_SLOW": 0x0571B0,
}


_WRAPPER_ASSIGNMENTS = tuple(
    attr
    for attr in functools.WRAPPER_ASSIGNMENTS
    # Skip __doc__ because we assign it on class creation using exec_body
    # callable that updates the namespace of the class.
    # Skip __annotations__ because there are differences between Python
    # versions on how it is initialized for a class that doesn't explicitly
    # define it and we don't want to force eager evaluation of anything that
    # would normally be lazy (mostly for consistency, shouldn't cause any
    # significant issues).
    if attr not in ("__annotations__", "__doc__")
)


def callers_module_name():
    # Call f_back twice since this function adds an extra frame
    return inspect.currentframe().f_back.f_back.f_globals["__name__"]


class _State(IntEnum):
    """Simple enum to track the type of wrapped object of a final proxy"""

    SLOW = 0
    FAST = 1


class _Unusable:
    """
    A totally unusable type. When a "fast" object is not available,
    it's useful to set it to _Unusable() so that any operations
    on it fail, and ensure fallback to the corresponding
    "slow" object.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError(
            "Fast implementation not available. "
            "Falling back to the slow implementation"
        )

    def __getattribute__(self, name: str) -> Any:
        if name in {"__class__"}:  # needed for type introspection
            return super().__getattribute__(name)
        raise TypeError("Unusable type. Falling back to the slow object")


class _PickleConstructor:
    """A pickleable object to support construction in __reduce__.

    This object is used to avoid having unpickling call __init__ on the
    objects, instead only invoking __new__. __init__ may have required
    arguments or otherwise perform invalid initialization that we could skip
    altogether since we're going to overwrite the wrapped object.
    """

    def __init__(self, type_):
        self._type = type_

    def __call__(self):
        return object.__new__(self._type)


def make_final_proxy_type(
    name: str,
    fast_type: type,
    slow_type: type,
    *,
    fast_to_slow: Callable,
    slow_to_fast: Callable,
    module: Optional[str] = None,
    additional_attributes: Mapping[str, Any] | None = None,
    postprocess: Callable[[_FinalProxy, Any, Any], Any] | None = None,
    bases: Tuple = (),
) -> Type[_FinalProxy]:
    """
    Defines a fast-slow proxy type for a pair of "final" fast and slow
    types. Final types are types for which known operations exist for
    converting an object of "fast" type to "slow" and vice-versa.

    Parameters
    ----------
    name: str
        The name of the class returned
    fast_type: type
    slow_type: type
    fast_to_slow: callable
        Function that accepts a single argument of type `fast_type`
        and returns an object of type `slow_type`
    slow_to_fast: callable
        Function that accepts a single argument of type `slow_type`
        and returns an object of type `fast_type`
    additional_attributes
        Mapping of additional attributes to add to the class (optional)
    postprocess
        Optional function called to allow the proxy to postprocess
        itself when being wrapped up, called with the proxy object,
        the unwrapped result object, and the function that was used to
        construct said unwrapped object. See also `_maybe_wrap_result`.

    Notes
    -----
    As a side-effect, this function adds `fast_type` and `slow_type`
    to a global mapping of final types to their corresponding proxy
    types, accessible via `get_final_type_map()`.
    """

    def __init__(self, *args, **kwargs):
        _fast_slow_function_call(
            lambda cls, *args, **kwargs: setattr(
                self, "_xdf_wrapped", cls(*args, **kwargs)
            ),
            type(self),
            *args,
            **kwargs,
        )

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_XDF_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="xdf_python",
    )
    def _xdf_slow_to_fast(self):
        # if we are wrapping a slow object,
        # convert it to a fast one
        if self._xdf_state is _State.SLOW:
            return slow_to_fast(self._xdf_wrapped)
        return self._xdf_wrapped

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_XDF_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="xdf_python",
    )
    def _xdf_fast_to_slow(self):
        # if we are wrapping a fast object,
        # convert it to a slow one
        if self._xdf_state is _State.FAST:
            return fast_to_slow(self._xdf_wrapped)
        return self._xdf_wrapped

    @property  # type: ignore
    def _xdf_state(self) -> _State:
        return (
            _State.FAST
            if isinstance(self._xdf_wrapped, self._xdf_fast_type)
            else _State.SLOW
        )

    def __reduce__(self):
        # Need a local import to avoid circular import issues
        from .module_finder import disable_transparent_mode_if_enabled

        with disable_transparent_mode_if_enabled():
            pickled_wrapped_obj = pickle.dumps(self._xdf_wrapped)
        return (_PickleConstructor(type(self)), (), pickled_wrapped_obj)

    def __setstate__(self, state):
        # Need a local import to avoid circular import issues
        from .module_finder import disable_transparent_mode_if_enabled

        with disable_transparent_mode_if_enabled():
            unpickled_wrapped_obj = pickle.loads(state)
        self._xdf_wrapped = unpickled_wrapped_obj

    slow_dir = dir(slow_type)
    cls_dict = {
        "__init__": __init__,
        "__doc__": inspect.getdoc(slow_type),
        "_xdf_slow_dir": slow_dir,
        "_xdf_fast_type": fast_type,
        "_xdf_slow_type": slow_type,
        "_xdf_slow_to_fast": _xdf_slow_to_fast,
        "_xdf_fast_to_slow": _xdf_fast_to_slow,
        "_xdf_state": _xdf_state,
        "__reduce__": __reduce__,
        "__setstate__": __setstate__,
    }
    if additional_attributes is None:
        additional_attributes = {}
    if overlap := (set(cls_dict) & set(additional_attributes)):
        raise RuntimeError(
            f"Some additional attributes ({overlap}) overlap with reserved "
            "names"
        )

    for method in _SPECIAL_METHODS:
        if getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)
    cls_dict.update(additional_attributes)

    cls = types.new_class(
        name,
        (*bases, _FinalProxy),
        {"metaclass": _FastSlowProxyMeta},
        lambda ns: ns.update(cls_dict),
    )
    functools.update_wrapper(
        cls,
        slow_type,
        assigned=_WRAPPER_ASSIGNMENTS,
        updated=(),
    )
    cls.__module__ = module if module is not None else callers_module_name()

    final_type_map = get_final_type_map()
    if fast_type is not _Unusable:
        final_type_map[fast_type] = cls
    final_type_map[slow_type] = cls

    return cls


def make_intermediate_proxy_type(
    name: str,
    fast_type: type,
    slow_type: type,
    *,
    module: Optional[str] = None,
) -> Type[_IntermediateProxy]:
    """
    Defines a proxy type for a pair of "intermediate" fast and slow
    types. Intermediate types are the types of the results of
    operations invoked on final types.

    As a side-effect, this function adds `fast_type` and `slow_type`
    to a global mapping of intermediate types to their corresponding
    proxy types, accessible via `get_intermediate_type_map()`.

    Parameters
    ----------
    name: str
        The name of the class returned
    fast_type: type
    slow_type: type
    """

    def __init__(self, *args, **kwargs):
        # disallow __init__. An intermediate proxy type can only be
        # instantiated from (possibly chained) operations on a final
        # proxy type.
        raise TypeError(
            f"Cannot directly instantiate object of type {type(self)}"
        )

    @property  # type: ignore
    def _xdf_state(self):
        return (
            _State.FAST
            if isinstance(self._xdf_wrapped, self._xdf_fast_type)
            else _State.SLOW
        )

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_XDF_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="xdf_python",
    )
    def _xdf_slow_to_fast(self):
        if self._xdf_state is _State.SLOW:
            return super(type(self), self)._xdf_slow_to_fast()
        return self._xdf_wrapped

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_XDF_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="xdf_python",
    )
    def _xdf_fast_to_slow(self):
        if self._xdf_state is _State.FAST:
            return super(type(self), self)._xdf_fast_to_slow()
        return self._xdf_wrapped

    slow_dir = dir(slow_type)
    cls_dict = {
        "__init__": __init__,
        "__doc__": inspect.getdoc(slow_type),
        "_xdf_slow_dir": slow_dir,
        "_xdf_fast_type": fast_type,
        "_xdf_slow_type": slow_type,
        "_xdf_slow_to_fast": _xdf_slow_to_fast,
        "_xdf_fast_to_slow": _xdf_fast_to_slow,
        "_xdf_state": _xdf_state,
    }

    for method in _SPECIAL_METHODS:
        if getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)

    cls = types.new_class(
        name,
        (_IntermediateProxy,),
        {"metaclass": _FastSlowProxyMeta},
        lambda ns: ns.update(cls_dict),
    )
    functools.update_wrapper(
        cls,
        slow_type,
        assigned=_WRAPPER_ASSIGNMENTS,
        updated=(),
    )
    cls.__module__ = module if module is not None else callers_module_name()

    intermediate_type_map = get_intermediate_type_map()
    if fast_type is not _Unusable:
        intermediate_type_map[fast_type] = cls
    intermediate_type_map[slow_type] = cls

    return cls


def register_proxy_func(slow_func: Callable):
    """
    Decorator to register custom function as a proxy for slow_func.

    Parameters
    ----------
    slow_func: Callable
        The function to register a wrapper for.

    Returns
    -------
    Callable
    """

    def wrapper(func):
        registered_functions = get_registered_functions()
        registered_functions[slow_func] = func
        functools.update_wrapper(func, slow_func)
        return func

    return wrapper


@functools.lru_cache(maxsize=None)
def get_final_type_map():
    """
    Return the mapping of all known fast and slow final types to their
    corresponding proxy types.
    """
    return _DictOfTypes()


@functools.lru_cache(maxsize=None)
def get_intermediate_type_map():
    """
    Return a mapping of all known fast and slow intermediate types to their
    corresponding proxy types.
    """
    return _DictOfTypes()


@functools.lru_cache(maxsize=None)
def get_registered_functions():
    return dict()


def _raise_attribute_error(obj, name):
    """
    Raise an AttributeError with a message that is consistent with
    the error raised by Python for a non-existent attribute on a
    proxy object.
    """
    raise AttributeError(f"'{obj}' object has no attribute '{name}'")


class _FastSlowAttribute:
    """
    A descriptor type used to define attributes of fast-slow proxies.
    """

    def __init__(self, name: str):
        self._name = name

    def __get__(self, obj, owner=None) -> Any:
        if obj is None:
            # class attribute
            obj = owner

        if not (
            isinstance(obj, _FastSlowProxy)
            or issubclass(type(obj), _FastSlowProxyMeta)
        ):
            # we only want to look up attributes on the underlying
            # fast/slow objects for instances of _FastSlowProxy or
            # subtypes of _FastSlowProxyMeta:
            _raise_attribute_error(owner if owner else obj, self._name)

        result, _ = _fast_slow_function_call(getattr, obj, self._name)

        if isinstance(result, functools.cached_property):
            # TODO: temporary workaround until dask is able
            # to correctly inspect cached_property objects.
            # GH: 264
            result = property(result.func)

        if isinstance(result, (_MethodProxy, property)):
            from .module_finder import disable_transparent_mode_if_enabled

            type_ = owner if owner else type(obj)
            slow_result_type = getattr(type_._xdf_slow, self._name)
            with disable_transparent_mode_if_enabled():
                result.__doc__ = inspect.getdoc(  # type: ignore
                    slow_result_type
                )

            if isinstance(result, _MethodProxy):
                # Note that this will produce the wrong result for bound
                # methods because dir for the method won't be the same as for
                # the pure unbound function, but the alternative is
                # materializing the slow object when we don't really want to.
                result._xdf_slow_dir = dir(slow_result_type)  # type: ignore

        return result


class _FastSlowProxyMeta(type):
    """
    Metaclass used to dynamically find class attributes and
    classmethods of fast-slow proxy types.
    """

    @property
    def _xdf_slow(self) -> type:
        return self._xdf_slow_type

    @property
    def _xdf_fast(self) -> type:
        return self._xdf_fast_type

    def __dir__(self):
        # Try to return the cached dir of the slow object, but if it
        # doesn't exist, fall back to the default implementation.
        try:
            return self._xdf_slow_dir
        except AttributeError:
            return type.__dir__(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_xdf") or name.startswith("__"):
            # an AttributeError was raised when trying to evaluate
            # an internal attribute, we just need to propagate this
            _raise_attribute_error(self.__class__.__name__, name)

        attr = _FastSlowAttribute(name)
        return attr.__get__(None, owner=self)


class _FastSlowProxy:
    """
    Base class for all fast=slow proxy types.

    A fast-slow proxy is proxy for a pair of types that provide "fast"
    and "slow" implementations of the same API.  At any time, a
    fast-slow proxy wraps an object of either "fast" type, or "slow"
    type. Operations invoked on the fast-slow proxy are first
    delegated to the "fast" type, and if that fails, to the "slow"
    type.
    """

    _xdf_wrapped: Any

    def _xdf_fast_to_slow(self) -> Any:
        """
        If the wrapped object is of "fast" type, returns the
        corresponding "slow" object. Otherwise, returns the wrapped
        object as-is.
        """
        raise NotImplementedError("Abstract base class")

    def _xdf_slow_to_fast(self) -> Any:
        """
        If the wrapped object is of "slow" type, returns the
        corresponding "fast" object. Otherwise, returns the wrapped
        object as-is.
        """
        raise NotImplementedError("Abstract base class")

    @property
    def _xdf_fast(self) -> Any:
        """
        Returns the wrapped object. If the wrapped object is of "slow"
        type, replaces it with the corresponding "fast" object before
        returning it.
        """
        self._xdf_wrapped = self._xdf_slow_to_fast()
        return self._xdf_wrapped

    @property
    def _xdf_slow(self) -> Any:
        """
        Returns the wrapped object. If the wrapped object is of "fast"
        type, replaces it with the corresponding "slow" object before
        returning it.
        """
        self._xdf_wrapped = self._xdf_fast_to_slow()
        return self._xdf_wrapped

    def __dir__(self):
        # Try to return the cached dir of the slow object, but if it
        # doesn't exist, fall back to the default implementation.
        try:
            return self._xdf_slow_dir
        except AttributeError:
            return object.__dir__(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_xdf"):
            # an AttributeError was raised when trying to evaluate
            # an internal attribute, we just need to propagate this
            _raise_attribute_error(self.__class__.__name__, name)
        if name in {
            "_ipython_canary_method_should_not_exist_",
            "_ipython_display_",
            "_repr_mimebundle_",
            # Workaround for https://github.com/numpy/numpy/issues/5350
            # see GH:216 for details
            "__array_struct__",
        }:
            # IPython always looks for these names in its display
            # logic. See #GH:70 and #GH:172 for more details but the
            # gist is that not raising an AttributeError immediately
            # results in slow display in IPython (since the fast
            # object will be copied to the slow one to look for
            # attributes there which then also won't exist).
            # This is somewhat delicate to the order in which IPython
            # implements special display fallbacks.
            _raise_attribute_error(self.__class__.__name__, name)
        if name.startswith("_"):
            # private attributes always come from `._xdf_slow`:
            return getattr(self._xdf_slow, name)
        attr = _FastSlowAttribute(name)
        return attr.__get__(self)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        return _FastSlowAttribute("__setattr__").__get__(self)(name, value)

    def __add__(self, other):
        return _fast_slow_function_call(operator.add, self, other)[0]

    def __radd__(self, other):
        return _fast_slow_function_call(operator.add, other, self)[0]

    def __sub__(self, other):
        return _fast_slow_function_call(operator.sub, self, other)[0]

    def __rsub__(self, other):
        return _fast_slow_function_call(operator.sub, other, self)[0]

    def __mul__(self, other):
        return _fast_slow_function_call(operator.mul, self, other)[0]

    def __rmul__(self, other):
        return _fast_slow_function_call(operator.mul, other, self)[0]

    def __truediv__(self, other):
        return _fast_slow_function_call(operator.truediv, self, other)[0]

    def __rtruediv__(self, other):
        return _fast_slow_function_call(operator.truediv, other, self)[0]

    def __floordiv__(self, other):
        return _fast_slow_function_call(operator.floordiv, self, other)[0]

    def __rfloordiv__(self, other):
        return _fast_slow_function_call(operator.floordiv, other, self)[0]

    def __mod__(self, other):
        return _fast_slow_function_call(operator.mod, self, other)[0]

    def __rmod__(self, other):
        return _fast_slow_function_call(operator.mod, other, self)[0]

    def __divmod__(self, other):
        return _fast_slow_function_call(divmod, self, other)[0]

    def __rdivmod__(self, other):
        return _fast_slow_function_call(divmod, other, self)[0]

    def __pow__(self, other):
        return _fast_slow_function_call(operator.pow, self, other)[0]

    def __rpow__(self, other):
        return _fast_slow_function_call(operator.pow, other, self)[0]

    def __lshift__(self, other):
        return _fast_slow_function_call(operator.lshift, self, other)[0]

    def __rlshift__(self, other):
        return _fast_slow_function_call(operator.lshift, other, self)[0]

    def __rshift__(self, other):
        return _fast_slow_function_call(operator.rshift, self, other)[0]

    def __rrshift__(self, other):
        return _fast_slow_function_call(operator.rshift, other, self)[0]

    def __and__(self, other):
        return _fast_slow_function_call(operator.and_, self, other)[0]

    def __rand__(self, other):
        return _fast_slow_function_call(operator.and_, other, self)[0]

    def __xor__(self, other):
        return _fast_slow_function_call(operator.xor, self, other)[0]

    def __rxor__(self, other):
        return _fast_slow_function_call(operator.xor, other, self)[0]

    def __or__(self, other):
        return _fast_slow_function_call(operator.or_, self, other)[0]

    def __ror__(self, other):
        return _fast_slow_function_call(operator.or_, other, self)[0]

    def __matmul__(self, other):
        return _fast_slow_function_call(operator.matmul, self, other)[0]

    def __rmatmul__(self, other):
        return _fast_slow_function_call(operator.matmul, other, self)[0]


class _FinalProxy(_FastSlowProxy):
    """
    Proxy type for a pair of fast and slow "final" types for which
    there is a known conversion from fast to slow, and vice-versa.
    The conversion between fast and slow types is done using
    user-provided conversion functions.

    Do not attempt to use this class directly. Instead, use
    `make_final_proxy_type` to create subtypes.
    """

    @classmethod
    def _xdf_wrap(cls, value, func):
        """Default mechanism to wrap a value in a proxy type

        Parameters
        ----------
        cls
            The proxy type
        value
            The value to wrap up
        func
            The function called that constructed value

        Returns
        -------
        A new proxied object

        Notes
        -----
        _FinalProxy subclasses can override this classmethod if they
        need particular behaviour when wrapped up.
        """
        proxy = object.__new__(cls)
        proxy._xdf_wrapped = value
        return proxy


class _IntermediateProxy(_FastSlowProxy):
    """
    Proxy type for a pair of "intermediate" types that appear as
    intermediate values when invoking operations on "final" types.
    The conversion between fast and slow types is done by keeping
    track of the sequence of operations that created the wrapped
    object, and "playing back" that sequence starting from the "slow"
    version of the originating _FinalProxy.

    Do not attempt to use this class directly. Instead, use
    `make_intermediate_proxy_type` to create subtypes.
    """

    _method_chain: Tuple[Callable, Tuple, Dict]

    @classmethod
    def _xdf_wrap(
        cls,
        obj: Any,
        method_chain: Tuple[Callable, Tuple, Dict],
    ):
        """
        Parameters
        ----------
        obj: The object to wrap
        method_chain: A tuple of the form (func, args, kwargs) where
            `func` is the function that was called to create `obj`,
            and `args` and `kwargs` are the arguments that were passed
            to `func`.
        """
        proxy = object.__new__(cls)
        proxy._xdf_wrapped = obj
        proxy._method_chain = method_chain
        return proxy

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_XDF_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="xdf_python",
    )
    def _xdf_slow_to_fast(self) -> Any:
        func, args, kwargs = self._method_chain
        args, kwargs = _fast_arg(args), _fast_arg(kwargs)
        return func(*args, **kwargs)

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_XDF_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="xdf_python",
    )
    def _xdf_fast_to_slow(self) -> Any:
        func, args, kwargs = self._method_chain
        args, kwargs = _slow_arg(args), _slow_arg(kwargs)
        return func(*args, **kwargs)


class _CallableProxyMixin:
    """
    Mixin class that implements __call__ for fast-slow proxies.
    """

    # For wrapped callables isinstance(self, FunctionType) should return True
    __class__ = types.FunctionType  # type: ignore

    def __call__(self, *args, **kwargs) -> Any:
        result, _ = _fast_slow_function_call(
            # We cannot directly call self here because we need it to be
            # converted into either the fast or slow object (by
            # _fast_slow_function_call) to avoid infinite recursion.
            # TODO: When Python 3.11 is the minimum supported Python version
            # this can use operator.call
            lambda x, *args, **kwargs: x(*args, **kwargs),
            self,
            *args,
            **kwargs,
        )
        return result


class _FunctionProxy(_CallableProxyMixin):
    """
    Proxy for a pair of fast and slow functions.
    """

    def __init__(self, fast: Callable | _Unusable, slow: Callable):
        self._xdf_fast = fast
        self._xdf_slow = slow
        functools.update_wrapper(self, slow)


class _MethodProxy(_CallableProxyMixin, _IntermediateProxy):
    """
    Methods of fast-slow proxies are of type _MethodProxy.
    """


def _fast_slow_function_call(func: Callable, /, *args, **kwargs) -> Any:
    """
    Call `func` with all `args` and `kwargs` converted to their
    respective fast type. If that fails, call `func` with all
    `args` and `kwargs` converted to their slow type.

    Wrap the result in a fast-slow proxy if it is a type we know how
    to wrap.
    """
    fast = False
    try:
        with nvtx.annotate(
            "EXECUTE_FAST",
            color=_XDF_NVTX_COLORS["EXECUTE_FAST"],
            domain="xdf_python",
        ):
            fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
            result = func(*fast_args, **fast_kwargs)
            if result is NotImplemented:
                # try slow path
                raise Exception()
            fast = True
    except Exception:
        with nvtx.annotate(
            "EXECUTE_SLOW",
            color=_XDF_NVTX_COLORS["EXECUTE_SLOW"],
            domain="xdf_python",
        ):
            slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
            result = func(*slow_args, **slow_kwargs)
    return _maybe_wrap_result(result, func, *args, **kwargs), fast


def _transform_arg(
    arg: Any, attribute_name: Literal["_xdf_slow", "_xdf_fast"], seen: Set[int]
) -> Any:
    """
    Transform "arg" into its corresponding slow (or fast) type.
    """
    import numpy as np

    if isinstance(arg, (_FastSlowProxy, _FastSlowProxyMeta, _FunctionProxy)):
        typ = getattr(arg, attribute_name)
        if typ is _Unusable:
            raise Exception("Cannot transform _Unusable")
        return typ
    elif isinstance(arg, types.ModuleType) and attribute_name in arg.__dict__:
        return arg.__dict__[attribute_name]
    elif isinstance(arg, (list, tuple)):
        transformed = (_transform_arg(a, attribute_name, seen) for a in arg)
        if hasattr(arg, "_make"):
            # namedtuple
            return type(arg)._make(transformed)
        return type(arg)(transformed)
    elif isinstance(arg, dict):
        return {
            _transform_arg(k, attribute_name, seen): _transform_arg(
                a, attribute_name, seen
            )
            for k, a in arg.items()
        }
    elif isinstance(arg, np.ndarray) and arg.dtype == "O":
        return np.asarray(
            [_transform_arg(a, attribute_name, seen) for a in arg.flat],
            dtype="O",
        ).reshape(arg.shape)
    elif isinstance(arg, Iterator) and attribute_name == "_xdf_fast":
        # this may include consumable objects like generators or
        # IOBase objects, which we don't want unavailable to the slow
        # path in case of fallback. So, we raise here and ensure the
        # slow path is taken:
        raise Exception()
    elif isinstance(arg, types.FunctionType):
        if id(arg) in seen:
            # `arg` is mutually recursive with another function.  We
            # can't handle these cases yet:
            return arg
        seen.add(id(arg))
        return _replace_closurevars(arg, attribute_name, seen)
    else:
        return arg


def _fast_arg(arg: Any) -> Any:
    """
    Transform "arg" into its corresponding fast type.
    """
    seen: Set[int] = set()
    return _transform_arg(arg, "_xdf_fast", seen)


def _slow_arg(arg: Any) -> Any:
    """
    Transform "arg" into its corresponding slow type.
    """
    seen: Set[int] = set()
    return _transform_arg(arg, "_xdf_slow", seen)


def _maybe_wrap_result(result: Any, func: Callable, /, *args, **kwargs) -> Any:
    """
    Wraps "result" in a fast-slow proxy if is a "proxiable" object.
    """
    if _is_final_type(result):
        typ = get_final_type_map()[type(result)]
        return typ._xdf_wrap(result, func)
    elif _is_intermediate_type(result):
        typ = get_intermediate_type_map()[type(result)]
        return typ._xdf_wrap(result, method_chain=(func, args, kwargs))
    elif _is_final_class(result):
        return get_final_type_map()[result]
    elif isinstance(result, list):
        return type(result)(
            [
                _maybe_wrap_result(r, operator.getitem, result, i)
                for i, r in enumerate(result)
            ]
        )
    elif isinstance(result, tuple):
        wrapped = (
            _maybe_wrap_result(r, operator.getitem, result, i)
            for i, r in enumerate(result)
        )
        if hasattr(result, "_make"):
            # namedtuple
            return type(result)._make(wrapped)
        else:
            return type(result)(wrapped)
    elif isinstance(result, Iterator):
        return (_maybe_wrap_result(r, lambda x: x, r) for r in result)
    elif _is_function_or_method(result):
        return _MethodProxy._xdf_wrap(
            result, method_chain=(func, args, kwargs)
        )
    else:
        return result


def _is_final_type(result: Any) -> bool:
    return isinstance(result, tuple(get_final_type_map().keys()))


def _is_final_class(result: Any) -> bool:
    if not isinstance(result, type):
        return False
    return any(issubclass(result, k) for k in get_final_type_map().keys())


def _is_intermediate_type(result: Any) -> bool:
    return isinstance(result, tuple(get_intermediate_type_map().keys()))


def _is_function_or_method(obj: Any) -> bool:
    return isinstance(
        obj,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.WrapperDescriptorType,
            types.MethodWrapperType,
            types.MethodDescriptorType,
            types.BuiltinMethodType,
        ),
    )


class _DictOfTypes(dict):
    # a `dict` meant to be used with types as keys, accepting subtypes
    # for lookup
    def __missing__(self, key: type):
        # if `key` was not found, see if a superclass of `key` is
        # found:
        for k in self.keys():
            if issubclass(key, k):
                return self[k]
        else:
            raise KeyError(key)


def _replace_closurevars(
    f: types.FunctionType,
    attribute_name: Literal["_xdf_slow", "_xdf_fast"],
    seen: Set[int],
) -> types.FunctionType:
    """
    Return a copy of `f` with its closure variables replaced with
    their corresponding slow (or fast) types.
    """
    if f.__closure__:
        # GH #254: If empty cells are present - which can happen in
        # situations like when `f` is a method that invokes the
        # "empty" `super()` - the call to `getclosurevars` below will
        # fail.  For now, we just return `f` in this case.  If needed,
        # we can consider populating empty cells with a placeholder
        # value to allow the call to `getclosurevars` to succeed.
        if any(c == types.CellType() for c in f.__closure__):
            return f

    f_nonlocals, f_globals, f_builtins, _ = inspect.getclosurevars(f)

    g_globals = _transform_arg(f_globals, attribute_name, seen)
    g_nonlocals = _transform_arg(f_nonlocals, attribute_name, seen)

    # if none of the globals/nonlocals were transformed, we
    # can just return f:
    if all(f_globals[k] is g_globals[k] for k in f_globals) and all(
        g_nonlocals[k] is f_nonlocals[k] for k in f_nonlocals
    ):
        return f

    g_closure = tuple(types.CellType(val) for val in g_nonlocals.values())
    g_globals["__builtins__"] = f_builtins

    g = types.FunctionType(
        f.__code__,
        g_globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=g_closure,
    )
    g = functools.update_wrapper(
        g,
        f,
        assigned=functools.WRAPPER_ASSIGNMENTS + ("__kwdefaults__",),
    )
    return g


_SPECIAL_METHODS: Set[str] = {
    "__repr__",
    "__str__",
    "__len__",
    "__contains__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__getslice__",
    "__setslice__",
    "__delslice__",
    "__iter__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__neg__",
    "__invert__",
    "__abs__",
    "__round__",
    "__format__",
    "__bool__",
    "__float__",
    "__int__",
    "__complex__",
    "__enter__",
    "__exit__",
    "__next__",
    "__copy__",
    "__deepcopy__",
    "__dataframe__",
    # Added on a per-proxy basis
    # https://github.com/rapidsai/xdf/pull/306#pullrequestreview-1636155428
    # "__hash__",
}
