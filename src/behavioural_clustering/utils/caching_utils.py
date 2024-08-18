import hashlib
from typing import Any, Callable
from functools import wraps

class CacheManager:
    def __init__(self):
        self.cache = {}

    def get_cache_key(self, prefix: str, *args: Any) -> str:
        """Generate a unique cache key based on the input parameters."""
        key = f"{prefix}_{hashlib.md5(str(args).encode()).hexdigest()}"
        return key

    def cached(self, prefix: str) -> Callable:
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = self.get_cache_key(prefix, *args, *kwargs.values())
                if cache_key not in self.cache:
                    self.cache[cache_key] = func(*args, **kwargs)
                return self.cache[cache_key]
            return wrapper
        return decorator

cache_manager = CacheManager()