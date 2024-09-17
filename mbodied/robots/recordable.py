import threading
import time
from queue import Queue
from typing import Callable, Literal, Self

from mbodied.data.recording import Recorder


class Recorder:
    """A class for non-blocking recording robot observation at a specified frequency.

    Robot class must pass in the `get_state`, `get_observation`, `prepare_action` methods.
    - `get_state()` gets the current state/pose of the robot.
    - `get_observation()` captures the observation/image of the robot.
    - `prepare_action()` calculates the action between the new and old states.
    """

    def __init__(
        self,
        frequency_hz: int = 5,
        recorder_kwargs: dict = None,
        on_static: Literal["record", "omit"] = "omit",
    ) -> None:
        """Initializes the RobotRecorder.

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
        """
        if recorder_kwargs is None:
            recorder_kwargs = {}
        self._recorder = Recorder(**recorder_kwargs)
        self.task = None

        self.last_recorded_state = None
        self.last_image = None

        self.recording = False
        self.frequency_hz = frequency_hz
        self.record_on_static = on_static == "record"
        self.recording_queue = Queue()

        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def __enter__(self):
        """Enter the context manager, starting the recording."""
        self.start_recording(self.task)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager, stopping the recording."""
        self.stop_recording()

    def record(self, task: str) -> Self:
        """Set the task and return the context manager."""
        self.task = task
        return self

    def reset(self) -> None:
        """Reset the recorder."""
        while self.recording:
            time.sleep(0.1)
        self._recorder.reset()

    def record_from_robot(self) -> None:
        """Records the current pose and captures an image at the specified frequency."""
        while self.recording:
            start_time = time.perf_counter()
            self.record_state()
            elapsed_time = time.perf_counter() - start_time
            # Sleep for the remaining time to maintain the desired frequency
            sleep_time = max(0, (1.0 / self.frequency_hz) - elapsed_time)
            time.sleep(sleep_time)

    def on_action_start(self, task: str = "") -> None:
        """Starts the recording of pose and image."""
        if not self.recording:
            self.task = task
            self.recording = True
            self.recording_thread = threading.Thread(target=self.record_from_robot)
            self.recording_thread.start()

    def stop_recording(self) -> None:
        """Stops the recording of pose and image."""
        if self.recording:
            self.recording = False
            self.recording_thread.join()

    def _process_queue(self) -> None:
        """Processes the recording queue asynchronously."""
        while True:
            image, instruction, action, state = self.recording_queue.get()
            self.recorder.record(observation={"image": image, "instruction": instruction}, action=action, state=state)
            self.recording_queue.task_done()

    def record_state(self) -> None:
        """Records the current pose and image if the pose has changed."""
        state = self.get_state()
        image = self.get_observation()

        # This is the beginning of the episode
        if self.last_recorded_state is None:
            self.last_recorded_state = state
            self.last_image = image
            return

        if state != self.last_recorded_state or self.record_on_static:
            action = self.prepare_action(self.last_recorded_state, state)
            self.recording_queue.put(
                (
                    self.last_image,
                    self.task,
                    action,
                    self.last_recorded_state,
                ),
            )
            self.last_image = image
            self.last_recorded_state = state

    def finalize(self) -> None:
        """Records the final pose and image after the movement completes."""
        self.record_state()
