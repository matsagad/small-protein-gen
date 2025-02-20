from typing import Callable, Dict, TypeVar

T = TypeVar("T")


def register(name: str, registry: Dict[str, T]) -> Callable[[T], T]:

    def wrapper(obj: T) -> T:
        if name in registry:
            raise Exception(f"{name} is already registered!")
        registry[name] = obj
        return obj

    return wrapper
