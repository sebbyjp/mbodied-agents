import h5py
from _typeshed import Incomplete
from gymnasium import spaces
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import Image as Image
from typing import Any

def add_space_metadata(space, group) -> None: ...
def create_dataset_for_space_dict(space_dict: spaces.Dict, group: h5py.Group) -> None: ...
def copy_and_delete_old(filename) -> None: ...

class Recorder:
    """Records a dataset to an h5 file. Saves images defined to folder with _frames appended to the name stem.

    Example:
      ```
      # Define the observation and action spaces
      observation_space = spaces.Dict({
          'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
          'instruction': spaces.Discrete(10)
      })
      action_space = spaces.Dict({
          'gripper_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
          'gripper_action': spaces.Discrete(2)
      })

      state_space = spaces.Dict({
          'position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
          'velocity': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
      })

      # Create a recorder instance
      recorder = Recorder(name='test_recorder', observation_space=observation_space, action_space=action_space, state_space=state_space)

      # Generate some sample data
      num_steps = 10
      for i in range(num_steps):
          observation = {
              'image': np.ones((224, 224, 3), dtype=np.uint8),
              'instruction': i
          }
          action = {
              'gripper_position': np.zeros((3,), dtype=np.float32),
              'gripper_action': 1
          }
          state = {
              'position': np.random.rand(3).astype(np.float32),
              'velocity': np.random.rand(3).astype(np.float32)
          }
          recorder.record(observation, action, state=state)

      # Save the statistics
      recorder.save_stats()

      # Close the recorder
      recorder.close()

      # Assert that the HDF5 file and directories are created
      assert os.path.exists('test_recorder.h5')
      assert os.path.exists('test_recorder_frames')
      ```
    """
    out_dir: Incomplete
    frames_dir: Incomplete
    file: Incomplete
    name: Incomplete
    filename: Incomplete
    observation_space: Incomplete
    action_space: Incomplete
    state_space: Incomplete
    supervision_space: Incomplete
    image_keys_to_save: Incomplete
    index: int
    def __init__(self, name: str = 'dataset.h5', observation_space: spaces.Dict | str | None = None, action_space: spaces.Dict | str | None = None, state_space: spaces.Dict | str | None = None, supervision_space: spaces.Dict | str | None = None, out_dir: str = 'saved_datasets', image_keys_to_save: list = None) -> None:
        """Initialize the Recorder.

        Args:
          name (str): Name of the file.
          observation_space (spaces.Dict): Observation space.
          action_space (spaces.Dict): Action space.
          state_space (spaces.Dict): State space.
          supervision_space (spaces.Dict): Supervision space.
          out_dir (str, optional): Directory of the output file. Defaults to 'saved_datasets'.
          image_keys_to_save (list, optional): List of image keys to save. Defaults to ['image'].
        """
    def reset(self) -> None:
        """Reset the recorder."""
    def configure_root_spaces(self, **spaces: spaces.Dict):
        """Configure the root spaces.

        Args:
          observation_space (spaces.Dict): Observation space.
          action_space (spaces.Dict): Action space.
          state_space (spaces.Dict): State space.
          supervision_space (spaces.Dict): Supervision space.
        """
    def record_timestep(self, group: h5py.Group, sample: Any, index: int) -> None:
        """Record a timestep.

        Args:
          group (h5py.Group): Group to record to.
          sample (Any): Sample to record.
          index (int): Index to record at.
        """
    def record(self, observation: Any | None = None, action: Any | None = None, state: Any | None = None, supervision: Any | None = None) -> None:
        """Record a timestep.

        Args:
          observation (Any): Observation to record.
          action (Any): Action to record.
          state (Any): State to record.
          supervision (Any): Supervision to record.
        """
    def close(self) -> None:
        """Closes the Recorder and send the data if train_config is set."""
