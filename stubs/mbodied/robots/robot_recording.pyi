from _typeshed import Incomplete
from typing import Callable, Literal

class RobotRecorder:
    """A class for recording robot observation and actions.

    Recording at a specified frequency on the observation and action of a robot. It leverages a queue and a worker
    thread to handle the recording asynchronously, ensuring that the main operations of the
    robot are not blocked.

    Robot class must pass in the `get_state`, `get_observation`, `prepare_action` methods.`
    get_state() gets the current state/pose of the robot.
    get_observation() captures the observation/image of the robot.
    prepare_action() calculates the action between the new and old states.
    """
    recorder: Incomplete
    task: Incomplete
    last_recorded_state: Incomplete
    last_image: Incomplete
    recording: bool
    frequency_hz: Incomplete
    record_on_static: Incomplete
    recording_queue: Incomplete
    get_state: Incomplete
    get_observation: Incomplete
    prepare_action: Incomplete
    def __init__(self, get_state: Callable, get_observation: Callable, prepare_action: Callable, frequency_hz: int = 5, recorder_kwargs: dict = None, on_static: Literal['record', 'omit'] = 'omit') -> None:
        '''Initializes the RobotRecorder.

        This constructor sets up the recording mechanism on the given robot, including the recorder instance,
        recording frequency, and the asynchronous processing queue and worker thread. It also
        initializes attributes to track the last recorded pose and the current instruction.

        Args:
            get_state: A function that returns the current state of the robot.
            get_observation: A function that captures the observation/image of the robot.
            prepare_action: A function that calculates the action between the new and old states.
            frequency_hz: Frequency at which to record pose and image data (in Hz).
            recorder_kwargs: Keyword arguments to pass to the Recorder constructor.
            on_static: Whether to record on static poses or not. If "record", it will record when the robot is not moving.
        '''
    def __enter__(self) -> None:
        """Enter the context manager, starting the recording."""
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None:
        """Exit the context manager, stopping the recording."""
    def record(self, task: str) -> RobotRecorder:
        """Set the task and return the context manager."""
    def reset_recorder(self) -> None:
        """Reset the recorder."""
    def record_from_robot(self) -> None:
        """Records the current pose and captures an image at the specified frequency."""
    recording_thread: Incomplete
    def start_recording(self, task: str = '') -> None:
        """Starts the recording of pose and image."""
    def stop_recording(self) -> None:
        """Stops the recording of pose and image."""
    def record_current_state(self) -> None:
        """Records the current pose and image if the pose has changed."""
    def record_last_state(self) -> None:
        """Records the final pose and image after the movement completes."""
