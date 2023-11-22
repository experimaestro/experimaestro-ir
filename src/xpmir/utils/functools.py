from functools import lru_cache, partial

try:
    from functools import cache
except ImportError:

    def cache(user_function, /):
        """Cache function for python 3.8"""
        return lru_cache(maxsize=None)(user_function)


def partial_cache(func, *args, **kwargs):
    """Combines a lru_cache with a partial"""
    return cache(partial(func, *args, **kwargs))
