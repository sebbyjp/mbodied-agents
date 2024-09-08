from _typeshed import Incomplete
from mbodied.types.sample import Sample as Sample
from pydantic import ConfigDict
from typing import Any

MotionType: Incomplete

def MotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, motion_type: MotionType = 'UNSPECIFIED', **kwargs) -> Any:
    '''Field for a motion.

    Args:
        default: Default value for the field.
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
        motion_type: Type of the motion. Can be "UNSPECIFIED", "OTHER", "ABSOLUTE", "RELATIVE", "VELOCITY", "TORQUE".
    '''
def AbsoluteMotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs) -> Any:
    """Field for an absolute motion.

    This field is used to define the shape and bounds of an absolute motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
def RelativeMotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs) -> Any:
    """Field for a relative motion.

    This field is used to define the shape and bounds of a relative motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
def VelocityMotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs) -> Any:
    """Field for a velocity motion.

    This field is used to define the shape and bounds of a velocity motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
def TorqueMotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs) -> Any:
    """Field for a torque motion.

    This field is used to define the shape and bounds of a torque motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
def OtherMotionField(default: Any = ..., bounds: list[float] | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs) -> Any:
    """Field for an other motion.

    This field is used to define the shape and bounds of an other motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """

class Motion(Sample):
    """Base class for a motion."""
    model_config: ConfigDict
