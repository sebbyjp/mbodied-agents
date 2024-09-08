import abc
from abc import ABC, abstractmethod

class HardwareInterface(ABC, metaclass=abc.ABCMeta):
    """Abstract base class for hardware interfaces.

    This class provides a template for creating hardware interfaces that can
    control robots or other hardware devices.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the hardware interface.

        Args:
            kwargs: Additional arguments to pass to the hardware interface.
        """
    @abstractmethod
    def do(self, *args, **kwargs) -> None:
        """Executes motion.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
    async def async_do(self, *args, **kwargs) -> None:
        """Asynchronously executes motion.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
    def fetch(self, *args, **kwargs) -> None:
        """Fetches data from the hardware.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
    def capture(self, *args, **kwargs) -> None:
        """Captures continuous data from the hardware.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
