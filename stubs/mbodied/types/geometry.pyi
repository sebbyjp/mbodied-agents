from mbodied.types.motion import AbsoluteMotionField as AbsoluteMotionField, Motion as Motion, MotionField as MotionField, RelativeMotionField as RelativeMotionField

class Pose3D(Motion):
    """Action for a 2D+1 space representing x, y, and theta."""
    x: float
    y: float
    theta: float

class LocationAngle(Pose3D):
    """Alias for Pose3D. A 2D+1 space representing x, y, and theta."""

class Pose6D(Motion):
    """Movement for a 6D space representing x, y, z, roll, pitch, and yaw."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class Pose(Pose6D):
    """Alias for Pose6D. A movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

class AbsolutePose(Pose):
    """Absolute pose of the robot in 3D space."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class RelativePose(Pose):
    """Relative pose of the robot in 3D space."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
