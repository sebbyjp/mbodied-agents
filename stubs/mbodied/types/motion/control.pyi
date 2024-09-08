from mbodied.types.geometry import LocationAngle as LocationAngle, Pose as Pose, Pose6D as Pose6D
from mbodied.types.motion import Motion as Motion, MotionField as MotionField
from typing import Sequence

class JointControl(Motion):
    """Motion for joint control."""
    value: float
    def space(self): ...

class FullJointControl(Motion):
    """Full joint control."""
    joints: Sequence[JointControl] | list[float]
    names: Sequence[str] | list[float] | None
    def space(self): ...

class HandControl(Motion):
    """Action for a 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand."""
    pose: Pose6D
    grasp: JointControl

class HeadControl(Motion):
    tilt: JointControl
    pan: JointControl

class MobileSingleHandControl(Motion):
    """Control for a robot that can move its base in 2D space with a 6D EEF control + grasp."""
    base: LocationAngle | None
    hand: HandControl | None
    head: HeadControl | None

class MobileSingleArmControl(Motion):
    """Control for a robot that can move in 2D space with a single arm."""
    base: LocationAngle | None
    arm: FullJointControl | None
    head: HeadControl | None

class MobileBimanualArmControl(Motion):
    """Control for a robot that can move in 2D space with two arms."""
    base: LocationAngle | None
    left_arm: FullJointControl | None
    right_arm: FullJointControl | None
    head: HeadControl | None

class HumanoidControl(Motion):
    """Control for a humanoid robot."""
    left_arm: FullJointControl | None
    right_arm: FullJointControl | None
    left_leg: FullJointControl | None
    right_leg: FullJointControl | None
    head: HeadControl | None

class LocobotActionOrAnswer(MobileSingleHandControl):
    answer: str | None
    sleep: bool | None
    home: bool | None
