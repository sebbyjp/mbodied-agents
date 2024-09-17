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

import asyncio
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import Context

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
from pathlib import Path
from time import sleep, time
from typing import Any, Generator, Generic, Self, TypeVar

from aioitertools import 

from mbodied.agents.agent import Agent
from mbodied.robots.robot_recording import Recorder
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

ObsT, StateT, ActT, SupervisionT = (
    TypeVar("ObsT", bound=Sample),
    TypeVar("StateT", bound=Sample),
    TypeVar("ActionT", bound=Sample),
    TypeVar("SupervisionT"),
)

class State(Sample):
    chainmap: Context
    local_state: Sample
    shared_state: Sample




class Robot(Agent, Generic[ObsT, StateT, ActT, SupervisionT]):
    """Abstract base class for a robot hardware or sim interface.

    **High Level Methods:**
    - `act(state, action)`: Executes an action or sequence of actions and returns the new state.
    - `observe()`: Captures an Observation.
    - `sense()`: Infers the current state of the robot.
    - `execute(action)`: Executes a motion.
    - `step(action, state)`: Executes an action or sequence of actions and returns the new state.
    - `recording(task: str)`: Context manager to record a session and save the data.
    -

    **Hooks**
    - on_action_start(action, state): Hook to run before executing an action.
    - on_action_end(action, state): Hook to run after executing an action.
    - on_act_and_stream_resume(action, state): Hook to run when resuming act and stream.
    - on_act_and_stream_yield(action, state, new_state): Hook to run when yielding act and stream.
    - on_record_session_start(task, state): Hook to run when starting a recording session.
    - on_record_session_end(task, state): Hook to run when ending a recording session.
    """

    def __init__(self, frequency_hz: int = 5, recorder_kwargs: dict | None = None, current_state: StateT | None = None):
        """Initializes the SimRobot and sets up the robot arm."""
        self.frequency_hz = frequency_hz
        self.recorder_kwargs = recorder_kwargs
        self.current_state = current_state

    def step(self, actions: ActT | list[ActT], state: StateT | None = None, frequency_hz: int | None = None) -> StateT:
        """Executes an action or sequence of actions and returns the new state.

        Ensures uniform execution time for each action over duration if specified.

        Args:
            motion: The motion to execute, commonly referred to as an action.
            num_steps: The number of steps to divide the motion into.
        """
        if not isinstance(actions, Sample | list[Sample]):
            raise TypeError("Action must be a Sample or a list of Samples to determine the motion.")
        period = 1 / frequency_hz if frequency_hz else 0.05


        duration = ilen(always_iterable(actions)) * (1 / frequency_hz if frequency_hz else 0 )
        num_steps = duration * frequency_hz if duration else 1
        if isinstance(actions, list):
            for act in actions:
                while time.time() - tic < self.execution_time / num_steps:
                    pass

                self.state: Sample = self.do(act / num_steps)

            # Number of steps to divide the motion into
            step_motion = [value / num_steps for value in action.flatten()]
            for _ in range(num_steps):
                self.current_pos = [round(x + y, 5) for x, y in zip(self.current_pos, step_motion, strict=False)]
                time.sleep(sleep_duration)

            print("New position:", self.current_pos)  # noqa: T201

        return self.current_pos

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

class Robot(ABC):
    """Abstract base class for robot hardware interfaces.

    This class serves as a blueprint for creating interfaces to control robot hardware
    or other devices. It provides essential methods and guidelines to implement
    functionalities such as executing motions, capturing observations, and recording
    robot data, which can be used for training models.

    Key Features:
    - organize your hardware interface in a modular fashion
    - support asynchronous dataset creation

    **Recording Capabilities:**
    To enable data recording for model training, the following methods need implementation:
    - `get_observation()`: Captures the current observation/image of the robot.
    - `get_state()`: Retrieves the current state (pose) of the robot.
    - `prepare_action()`: Computes the action performed between two robot states.

    **Example Usage:**
    ```python
    robot = MyRobot()
    robot.init_recorder(frequency_hz=5, recorder_kwargs={...})
    with robot.record("pick up the remote"):
        robot.do(motion1)
        robot.do(motion2)
        ...
    ```
    When ``with robot.record`` is called, it starts recording the robot's observation and actions
    at the desired frequency. It stops when the context manager exits.

    Alternatively, you can manage recordings with `start_recording()` and `stop_recording()` methods.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initializes the robot hardware interface.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    @abstractmethod
    def do(self,  *args, state: Sample | Any | None = None, **kwargs) -> None:
        """Executes raw motion commands on the robot hardware. Lower-level than step or act.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    async def async_do(self, *args, **kwargs) -> None:
        """Asynchronously executes motion.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        return await asyncio.to_thread(self.do, *args, **kwargs)


    def act(self, *args, **kwargs) -> None:
        """Captures continuous data from the hardware asynchronously.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    def observe(self) -> Sample:
        """(Optional for robot recorder): Captures the observation/image of the robot.

        This will be used by the robot recorder to record the current observation/image of the robot.
        """
        raise NotImplementedError

    def infer(self) -> Sample:
        """Gets the current state of the robot.

        This will be used by the robot recorder to record the current state of the robot.
        """
        raise NotImplementedError

    # def prepare_action(self, old_state: Sample, new_state: Sample) -> Sample:
    #     """(Optional for robot recorder): Prepare the the action between two robot states.

    #     This is what you are recording as the action. For example, substract old from new hand position
    #     and use absolute value for grasp.

    #     Args:
    #         old_state: The old state (pose) of the robot.
    #         new_state: The new state (pose) of the robot.
    #     """
    #     raise NotImplementedError

    def init_recorder(
        self,
        frequency_hz: int = 5,
        recorder_kwargs: dict = None,
        on_static: str = "omit",
    ) -> None:
        """Initializes the recorder for the robot."""
        self._recorder = Recorder(
            self.get_state,
            self.get_observation,
            self.prepare_action,
            frequency_hz,
            recorder_kwargs,
            on_static=on_static,
        )

    @contextmanager
    def record(self, task: str) -> Generator[Self, None, None]:
        """Start recording with the given task with context manager.

        Usage:
        ```python
        with robot.record("pick up the remote"):
            robot.do(motion1)
            robot.do(motion2)
            ...
        ```
        """
        try:
            self._recorder.start_recording(task)
            yield self._recorder
        finally:
            self._recorder.stop_recording()
