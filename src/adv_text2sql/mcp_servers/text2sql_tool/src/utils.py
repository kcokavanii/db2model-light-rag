import logging
from functools import wraps

logger = logging.getLogger("text2sql_tool")


def print_result():
    """
    Декоратор: печатает результат работы функции, если включён debug.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(result)
            return result

        return wrapper

    return decorator
