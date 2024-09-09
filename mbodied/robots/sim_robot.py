# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from pathlib import Path
from typing import Generic, TypeVar

from mbodied.robots import Robot
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

ObsT, StaT, ActT, SupT = TypeVar("ObsT", bound=Sample), TypeVar("StateT", bound=Sample), TypeVar("ActionT", bound=Sample), TypeVar("SupT")

class MockRobot(Robot, Generic[ObsT, StaT, ActT, SupT]):
    """A simulated robot interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.
    do() simulates the execution of HandControl motions that got executed in execution_time.

    Attributes:
        home_state: The default state of the robot arm.
        current_state: The current state of the robot arm.
    """

    def __init__(self, execution_time: float = 1.0):
        """Initializes the SimRobot and sets up the robot arm.

        Args:
            execution_time: The time it takes to execute a motion.

        position: [x, y, z, r, p, y, grasp]
        """
        self.execution_time = execution_time
        self.home_pos = [0, 0, 0, 0, 0, 0, 0]
        self.current_pos = self.home_pos

    # def step(self, action: ActT | list[ActT], state: StaT | None = None,  duration: float | None = None) -> StaT:
    #     """Executes an action or sequence of actions and returns the new state.

    #     Ensures uniform execution time for each action over duration if specified.

    #     Args:
    #         motion: The motion to execute, commonly referred to as an action.
    #         num_steps: The number of steps to divide the motion into.
    #     """
    #     if not isinstance(action, Sample | list[Sample]):
    #         raise TypeError("Action must be a Sample or a list of Samples to determine the motion.")
    #     tic = time.time()
    #     if isinstance(action, list):
    #         for act in action:
    #             while time.time() - tic < self.execution_time / num_steps:
    #                 pass

    #             self.state: Sample = self.do(act / num_steps)

    #         # Number of steps to divide the motion into
    #         step_motion = [value / num_steps for value in action.flatten()]
    #         for _ in range(num_steps):
    #             self.current_pos = [round(x + y, 5) for x, y in zip(self.current_pos, step_motion, strict=False)]
    #             time.sleep(sleep_duration)

    #         print("New position:", self.current_pos)  # noqa: T201

    #     return self.current_pos

    def capture(self, **_) -> ObsT:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))

    def observe(self) -> ObsT:
        """Alias of capture for recording."""
        return self.capture()

    def infer(self) -> StaT:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return self.state


class MockImageHandRobot(MockRobot[Image, HandControl, HandControl, None]):
    """A simulated robot interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.
    do() simulates the execution of HandControl motions that got executed in execution_time.

    Attributes:
        home_state: The default state of the robot arm.
        current_state: The current state of the robot arm.
    """

    def __init__(self, execution_time: float = 1.0):
        """Initializes the SimRobot and sets up the robot arm.

        Args:
            execution_time: The time it takes to execute a motion.
        """
        super().__init__(execution_time)

    def capture(self, **_) -> ObsT:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))


    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses."""
        # Calculate the difference between the old and new poses. Use absolute value for grasp.
        old = list(old_pose.flatten())
        new = list(new_pose.flatten())
        result = [(new[i] - old[i]) for i in range(len(new) - 1)] + [new[-1]]
        return HandControl.unflatten(result)
