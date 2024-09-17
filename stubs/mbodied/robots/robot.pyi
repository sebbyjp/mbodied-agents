import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from mbodied.robots.recordable import RobotRecorder as RobotRecorder
from mbodied.types.sample import Sample as Sample

class Robot(ABC, metaclass=abc.ABCMeta):
    '''Abstract base class for robot hardware interfaces.

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
    When ``with robot.record`` is called, it starts recording the robot\'s observation and actions
    at the desired frequency. It stops when the context manager exits.

    Alternatively, you can manage recordings with `start_recording()` and `stop_recording()` methods.
    '''
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the robot hardware interface.

        Args:
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
    @abstractmethod
    def do(self, *args, **kwargs) -> None:
        """Executes motion.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
    async def async_do(self, *args, **kwargs) -> None:
        """Asynchronously executes motion.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
    def fetch(self, *args, **kwargs) -> None:
        """Fetches data from the hardware.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
    def capture(self, *args, **kwargs) -> None:
        """Captures continuous data from the hardware.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
    def get_observation(self) -> Sample:
        """(Optional for robot recorder): Captures the observation/image of the robot.

        This will be used by the robot recorder to record the current observation/image of the robot.
        """
    def get_state(self) -> Sample:
        """(Optional for robot recorder): Gets the current state (pose) of the robot.

        This will be used by the robot recorder to record the current state of the robot.
        """
    def prepare_action(self, old_state: Sample, new_state: Sample) -> Sample:
        """(Optional for robot recorder): Prepare the the action between two robot states.

        This is what you are recording as the action. For example, substract old from new hand position
        and use absolute value for grasp, etc.

        Args:
            old_state: The old state (pose) of the robot.
            new_state: The new state (pose) of the robot.
        """
    robot_recorder: Incomplete
    def init_recorder(self, frequency_hz: int = 5, recorder_kwargs: dict = None, on_static: str = 'omit') -> None:
        """Initializes the recorder for the robot."""
    def record(self, task: str) -> RobotRecorder:
        '''Start recording with the given task with context manager.

        Usage:
            with robot.record("pick up the remote"):
                robot.do(motion1)
                robot.do(motion2)
                ...
        '''
    def start_recording(self, task: str) -> None:
        """Start recording with the given task."""
    def stop_recording(self) -> None:
        """Stop recording."""
