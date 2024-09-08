# Provide backwards compatibility for old imports
import sys
import warnings
import embdata
from typing import Any, Callable

from embdata import episode, features, sense
from embdata.sample import Sample
from embdata.utils.import_utils import smart_import

# Import and re-export the to_features function
from embdata.features import to_features_dict

# Re-export the modules and functions


def getattr_migration(module_name: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2."""
    def wrapper(name: str) -> Any:
        if name == "__path__":
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            if name == 'features':
                return sys.modules[__name__]
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

__all__ = [module.__name__ for module in dir(embdata) if not module.startswith('_')]