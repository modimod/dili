import operator

from inspect import signature
from typing import Callable, Tuple


def function_accepts(*types) -> Callable:
    """
    Enforce function argument types (taken from PEP 318, slightly modified).

    :param types: a collection of types to support (one entry for each argument)
    :return: wrapped function
    """

    def check_accepts(f):
        assert len(types) == len(signature(f).parameters)

        def new_f(*args, **kwargs):
            for (a, t) in zip(args, types):
                try:
                    a = t(a)
                except TypeError as type_error:
                    raise TypeError(r'arg {0} does not match {1}'.format(repr(a), str(t))) from type_error
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def method_accepts(*types) -> Callable:
    """
    Enforce method argument types (taken from PEP 318, slightly modified).

    :param types: a collection of types to support (one entry for each argument)
    :return: wrapped method
    """

    def check_accepts(f):
        assert len(types) == len(signature(f).parameters) - 1

        def new_f(*args, **kwargs):
            for (a, t) in zip(args[1:], types):
                try:
                    a = t(a)
                except TypeError as type_error:
                    raise TypeError(r'arg {0} does not match {1}'.format(repr(a), str(t))) from type_error
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def function_bounds(*bounds, modes: Tuple[Tuple[callable, callable], ...] = ((operator.ge, operator.lt),)) -> Callable:
    """
    Enforce function argument boundaries.

    :param bounds: a collection of boundary-2-tuple (one entry for each argument)
    :param modes: a collection of comparator-2-tuple (one entry for each argument)
    :return: wrapped function
    """

    def check_accepts(f):
        assert len(bounds) == len(signature(f).parameters)
        _modes = (modes * (len(bounds) - len(modes) + 1))[:len(bounds)]

        def new_f(*args, **kwargs):
            for (a, t, m) in zip(args, bounds, _modes):
                if (a is not None) and not (m[0](a, t[0]) and m[1](a, t[1])):
                    raise ValueError(r'arg {0} is not in {1} with {2}'.format(repr(a), str(t), str(m)))
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def method_bounds(*bounds, modes: Tuple[Tuple[callable, callable], ...] = ((operator.ge, operator.lt),)) -> Callable:
    """
    Enforce method argument boundaries.

    :param bounds: a collection of boundary-2-tuple (one entry for each argument)
    :param modes: a collection of comparator-2-tuple (one entry for each argument)
    :return: wrapped method
    """

    def check_accepts(f):
        assert len(bounds) == len(signature(f).parameters) - 1
        _modes = (modes * (len(bounds) - len(modes) + 1))[:len(bounds)]

        def new_f(*args, **kwargs):
            for (a, t, m) in zip(args[1:], bounds, _modes):
                if (a is not None) and not (m[0](a, t[0]) and m[1](a, t[1])):
                    raise ValueError(r'arg {0} is not in {1} with {2}'.format(repr(a), str(t), str(m)))
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts
