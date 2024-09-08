import sys
from typing import Any, Callable
import embdata
from embdata.utils.import_utils import smart_import
from embdata.sample import Sample

def getattr_migration(module_name: str) -> Callable[[str, Any]]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2."""
    def wrapper(name: str) -> Any:
        """Raise an error if the object is not found, or warn if it was moved."""
        import warnings

        if name == "__path__":
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            if name not in sys.modules:
                import_path = f"embdata.{name}"
                imported_module = smart_import(import_path)
            else:
                imported_module = sys.modules[name]
        except ModuleNotFoundError as e:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from e
        else:
            warnings.warn(f"{module_name!r} was moved to embdata. Please update your imports.", DeprecationWarning, stacklevel=2)
            return imported_module

    return wrapper

# Import specific modules from embdata
from embdata import sense, episode, features

__all__ = ['Sample', 'sense', 'episode', 'features']
