import abc
from abc import abstractmethod
from typing import Generator

class Backend(metaclass=abc.ABCMeta):
    """Base class for agent backends."""
    @abstractmethod
    def predict(self, *args, **kwargs) -> str: ...
    @abstractmethod
    def stream(self, *args, **kwargs) -> Generator[str, None, None]: ...
