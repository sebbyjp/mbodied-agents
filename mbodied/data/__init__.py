# Provide backwards compatibility for old imports
import sys
import warnings
from typing import Any, Callable

import embdata.sample

# Import other necessary modules
from embdata import (
    coordinate,
    describe,
    episode,
    features,
    geometry,
    motion,
    ndarray,
    sample,
    sense,
    supervision,
    time,
    trajectory,
    units,
    utils,
)

# from embdata.coordinate import Coordinate, CoordinateField, PlanarPose, Plane, PlaneModel, Point, Pose, Pose3D, Pose6D
# from embdata.ndarray import NumpyArray
from embdata.sample import Sample

# from embdata.sense import camera, depth, image, state, world, zoe_depth
# from embdata.utils import image_utils, import_utils, iter_utils, schema_utils
from embdata.utils.import_utils import smart_import


def getattr_migration(module_name: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration to V2."""

    def wrapper(name: str) -> Any:
        print(f"module_name: {module_name}, name: {name}")
        try:
            import_path = f"embdata.{name}"
            imported_module = smart_import(import_path)
            globals()[name] = imported_module
            # for attr in dir(imported_module):
            #     globals()[attr] = getattr(imported_module, attr)
        except ModuleNotFoundError as e:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from e
        else:
            warnings.warn(
                f"{module_name!r} was moved to embdata. Please update your imports.", DeprecationWarning, stacklevel=2
            )
            return imported_module

    return wrapper


__getattr__ = getattr_migration(__name__)

__all__ = [
    "features",
    "episode",
    "sense",
    "supervision",
    "sample",
    "motion",
    "utils",
    "time",
    "units",
    "trajectory",
    "ndarray",
    "geometry",
    "describe",
    "coordinate",
    "Sample",
]
