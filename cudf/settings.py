"""
This module provides context based configuration of user options.

Available keys:

- formatting : Changes formatting behavior
    - nrows : max number of rows to show
    - ncols : max number of columns to show

"""
from copy import deepcopy
import threading
from contextlib import contextmanager
from collections import defaultdict


class _settings(object):
    """Wraps and manages the thread-local stack of configurations.
    Inner context inherit the settings from the parent.
    """
    tls = threading.local()

    def _make_stack_item(self):
        return defaultdict(dict)

    @property
    def _stack(self):
        """Get TLS stack"""
        tls = self.tls
        try:
            # Try getting the stack
            stack = tls.stack
        except AttributeError:
            # Make TLS stack on-demand
            tls.stack = stack = [self._make_stack_item()]
        return stack

    @property
    def _tos(self):
        """Get the top of stack"""
        return self._stack[-1]

    def __getattr__(self, name):
        return self._tos[name]

    def _push(self, **kwargs):
        # Copy previous context and update
        dct = deepcopy(self._tos)
        for k, v in kwargs.items():
            dct[k].update(v)
        # Add new context
        self._stack.append(dct)

    def _pop(self):
        self._stack.pop()

    @contextmanager
    def set_options(self, **kwargs):
        self._push(**kwargs)
        try:
            yield
        finally:
            self._pop()


# The global singleton for global settings
settings = _settings()
set_options = settings.set_options


# Singleton NOTSET class & object
class NOTSET(object):
    def __repr__(self):
        return "NOTSET"


NOTSET = NOTSET()
