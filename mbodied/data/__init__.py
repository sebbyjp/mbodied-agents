# Provide backwards compatibility for old imports
import sys
import warnings
from typing import Any, Callable

from embdata import episode, features, sense
from embdata.sample import Sample
from embdata.utils.import_utils import smart_import

# Re-export the modules and functions


# Import and re-export the to_features function
from embdata.features import to_features_dict


def getattr_migration(module_name: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2."""
    def wrapper(name: str) -> Any:
        if name == "__path__":
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            import_path = f"embdata.{name}"
            imported_module = smart_import(import_path)
        except ModuleNotFoundError as e:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from e
        else:
            warnings.warn(f"{module_name!r} was moved to embdata. Please update your imports.", DeprecationWarning, stacklevel=2)
            return imported_module

    return wrapper

__getattr__ = getattr_migration(__name__)

__all__ = [getattr(__name__, name) for name in dir(sys.modules[__name__]) if not name.startswith("_")]