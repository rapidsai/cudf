# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


def pipe(obj, func, *args, **kwargs):
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to the function or, in the case that the func is a tuple,
    interpret the first element of the tuple as a function and pass the obj to
    that function as a keyword argument whose key is the value of the second
    element of the tuple.

    Parameters
    ----------
    func : callable or tuple of (callable, str)
        Function to apply to this object or, alternatively, a
        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a
        string indicating the keyword of `callable`` that expects the
        object.
    *args : iterable, optional
        Positional arguments passed into ``func``.
    **kwargs : dict, optional
        A dictionary of keyword arguments passed into ``func``.

    Returns
    -------
    object : the return type of ``func``.
    """
    if isinstance(func, tuple):
        func, target = func
        if target in kwargs:
            raise ValueError(
                f"{target} is both the pipe target and a keyword argument"
            )
        kwargs[target] = obj
        return func(*args, **kwargs)
    else:
        return func(obj, *args, **kwargs)
