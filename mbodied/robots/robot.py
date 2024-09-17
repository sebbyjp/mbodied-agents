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
from aiostream.pipe import spaceout, cycle
from more_itertools import ilen, always_iterable, time_limited
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
from typing import Any, Callable, Generator, Generic, Self, TypeVar


from mbodied.agents.agent import Agent
from mbodied.robots.recordable import Recorder
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

    **Example Usage:**
    ```python
    robot = MyRobot()
    with robot.record("pick up the remote"):
        robot.do(motion1)
        robot.do(motion2)
        ...
    ```

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

    def _init_recorder(
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
    
    def step(self, actions: ActT | list[ActT], state: StateT | None = None, frequency_hz: int | None = None, duration: float | None = None, 
            stopping_predicate=Callable[[StateT, StateT], bool] | None) -> StateT:
        """Executes an action or sequence of actions and returns the new state. 
        
        If a single action is provided and no duration or stopping_predicate is specified, the action is executed once.
        Otherwise, the action is executed repeatedly until the duration is reached or the stopping_predicate is met.

        Ensures uniform execution time for each action over duration if specified.

        Args:
            motion: The motion to execute, commonly referred to as an action.
            num_steps: The number of steps to divide the motion into.
        """
        if not isinstance(actions, Sample | list[Sample]):
            raise TypeError("Action must be a Sample or a list of Samples to determine the motion.")
        period = 1 / frequency_hz if frequency_hz else 0.05

        if not isinstance(actions, list):
            actions = [actions]
            actions = cycle(actions) if not duration else actions * int(duration * frequency_hz)
        
        for action in spaceout(time_limited(cycle(always_iterable(actions), duration), period)):
            new_state = self.do(action)
            if stopping_predicate and stopping_predicate(state, new_state):
                break
            state = new_state

        return self.current_pos

    def do(self, action: ActT) -> StateT:
        """Executes a motion.

        Args:
            action: The motion to execute.
        """
        raise NotImplementedError

    def observe(self) -> ObsT:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))

    def sense(self) -> StateT:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return self.state
    
    def record_session(self, task: str) -> None:
        



    # def prepare_action(self, old_state: Sample, new_state: Sample) -> Sample:
    #     """(Optional for robot recorder): Prepare the the action between two robot states.

    #     This is what you are recording as the action. For example, substract old from new hand position
    #     and use absolute value for grasp.

    #     Args:
    #         old_state: The old state (pose) of the robot.
    #         new_state: The new state (pose) of the robot.
    #     """
    #     raise NotImplementedError

  

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
