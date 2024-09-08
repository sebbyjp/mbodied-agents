import importlib
import sys
from types import ModuleType

import embdata
import embdata.coordinate
import embdata.describe
import embdata.episode
import embdata.features
import embdata.geometry
import embdata.language
import embdata.motion
import embdata.sample
import embdata.sense
import embdata.time
import embdata.trajectory
import embdata.utils



if "mbodied.data" not in sys.modules:
    # Create a new module named 'mbodied.data'
    data = ModuleType("mbodied.data")
    sys.modules["mbodied.data"] = data
    sys.modules["mbodied.data"].__package__ = "mbodied"
    sys.modules["mbodied.data"].__name__ = "mbodied.data"
    sys.modules["mbodied.data"].__file__ = "data.py"
    # Set __path__ to simulate that 'mbodied.data' is a package
    data.__path__ = [__path__[0].replace("__init__.py", "data.py")]

    sys.modules["mbodied.data"] = data

for module in  [module for module in dir(embdata) if not module.startswith("__")] :
    if module in sys.modules["mbodied.data"].__dict__:
        continue
    # Check if 'mbodied.data' already exists in sys.modules
    sys.modules["mbodied.data"].__dict__[module] = getattr(embdata, module)
    sys.modules["mbodied.data"].__dict__[module].__package__ = "mbodied.data"
  
    setattr(data, module, getattr(embdata, module))
    globals()[module] = getattr(embdata, module)
    globals().update({f"data.{module}": getattr(embdata, module)})
    

data.__all__ = [module for module in dir(embdata) if not module.startswith("__")]

# Ensure 'data' is part of the exports
__all__ = [
    'data',
    "data.coordinate",
    "data.describe",
    "data.episode",
    "data.geometry",
    "data.language",
    "data.motion",
    "data.sample",
    "data.sense",
    "data.time",
    "data.trajectory",
    "data.utils",
    "data.features",
]


print(dir(data))