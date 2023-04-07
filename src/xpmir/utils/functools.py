from functools import lru_cache, partial


def cache(user_function, /):
    # PY_VERSION: 3.8
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    return lru_cache(maxsize=None)(user_function)


def partial_cache(func, *args, **kwargs):
    """Combines a lru_cache with a partial"""
    return cache(partial(func, *args, **kwargs))
