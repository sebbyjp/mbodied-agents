# This is a stub file for the conf module
# It should contain any necessary type information for the conf module

from typing import Any, Dict

def asdict() -> Dict[str, Any]: ...
def load() -> Any: ...

get: Any

# Add a note to clarify this is a stub file
__all__ = ['asdict', 'load', 'get']
