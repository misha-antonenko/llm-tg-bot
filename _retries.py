import asyncio as aio
import logging
from functools import wraps
from random import random


def retry(
    *retried_exc_classes: type[BaseException],
    excluded_exc_classes: tuple[type[BaseException], ...] = (),
    n_retries=12,
    backoff_factor=2.0,
    initial_delay=1.0,
    jitter_factor=0.25,
):
    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            delay = initial_delay
            for _ in range(n_retries):
                try:
                    return await func(*args, **kwargs)
                except excluded_exc_classes:
                    raise
                except retried_exc_classes as exc:
                    logging.warning(
                        'Retrying a call to %r in %.2f minutes after an exception:',
                        func.__name__,
                        delay / 60,
                        exc_info=exc,
                    )
                    await aio.sleep(delay * (1 + jitter_factor * (2 * random() - 1)))
                    delay *= backoff_factor
            return await func(*args, **kwargs)

        return wrapped

    return decorator
