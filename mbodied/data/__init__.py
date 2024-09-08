# Provide backwards compatibility for old imports
import sys
import warnings
from typing import Any, Callable

# Import and re-export the features module and its contents
from . import features
from .features import *

# Import other necessary modules
from embdata import episode, sense
from embdata.sample import Sample
from embdata.utils.import_utils import smart_import

def getattr_migration(module_name: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2."""
    def wrapper(name: str) -> Any:
        if name == "__path__":
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        if name in globals():
            return globals()[name]
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

# Expose to_features_dict as to_features for backward compatibility
from .features import to_features_dict as to_features

__all__ = ['features', 'to_features', 'episode', 'sense', 'Sample', 'smart_import']
