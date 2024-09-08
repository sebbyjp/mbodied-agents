import sys
from typing import Any, Callable
import embdata
from embdata.utils.import_utils import smart_import
import embdata.sample
import embdata.features
import embdata.utils
import embdata.describe
import embdata.episode
import embdata.sense


def getattr_migration(module_name: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2.

    Args:
        module_name: The module name.

    Returns:
        A callable that will raise an error if the object is not found.
    """
    from pydantic.errors import PydanticImportError

    def wrapper(name: str) -> object:
        """Raise an error if the object is not found, or warn if it was moved.

        In case it was moved, it still returns the object.

        Args:
            name: The object name.

        Returns:
            The object.
        """
        import warnings


        if name == "__path__":
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            if name not in sys.modules:
                import_path = f"{}:{name}"
                imported_module = smart_import(import_path)
        except ModuleNotFoundError as e:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from e
        else:
            warnings.warn(f"{module_name!r} was moved to embdata. Please update your imports.", DeprecationWarning)  # noqa: B028
            return imported_module

    globals: dict[str, Any] = sys.modules[module].__dict__
    if module in globals:
        return globals[module]

    return wrapper


from ..embdata import *

__all__ = [name for name in dir() if not name.startswith("_")]
