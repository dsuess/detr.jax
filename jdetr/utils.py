from typing import Optional, TypeVar

import haiku as hk
import jax


class Init:
    # pylint: disable=too-few-public-methods
    def __init__(self, module, *args, **kwargs):
        self._module = module
        self._args = args
        self._kwargs = kwargs

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        def func_runner(*args, **kwargs):
            def run_func(*args, **kwargs):
                instance = self._module(*self._args, **self._kwargs)
                return getattr(instance, name)(*args, **kwargs)

            func = hk.transform_with_state(run_func)
            params, state = func.init(jax.random.PRNGKey(0), *args, **kwargs)
            result, _ = func.apply(
                params, state, jax.random.PRNGKey(0), *args, **kwargs
            )
            return result

        return func_runner


T = TypeVar("T")


def maybe(value: Optional[T], default: T) -> T:
    return value if value is not None else default
