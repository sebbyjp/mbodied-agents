import gymnasium.spaces.dict
import h5py._hl.group
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import Image as Image
from typing import Any

def add_space_metadata(space, group) -> None: ...
def create_dataset_for_space_dict(space_dict: gymnasium.spaces.dict.Dict, group: h5py._hl.group.Group) -> None: ...
def copy_and_delete_old(filename) -> None: ...

class Recorder:
    def __init__(self, name=..., observation_space=..., action_space=..., state_space=..., supervision_space=..., out_dir=..., image_keys_to_save=...) -> None:
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
    def configure_root_spaces(self, **spaces: gymnasium.spaces.dict.Dict):
        """Configure the root spaces.

                Args:
                    **spaces: Spaces to configure.
                        observation_space (spaces.Dict): Observation space.
                        action_space (spaces.Dict): Action space.
                        state_space (spaces.Dict): State space.
                        supervision_space (spaces.Dict): Supervision space.
        """
    def record_timestep(self, group: h5py._hl.group.Group, sample: Any, index: int) -> None:
        """Record a timestep.

                Args:
                  group (h5py.Group): Group to record to.
                  sample (Any): Sample to record.
                  index (int): Index to record at.
        """
    def record(self, observation=..., action=..., state=..., supervision=...) -> None:
        """Record a timestep.

                Args:
                  observation (Any): Observation to record.
                  action (Any): Action to record.
                  state (Any): State to record.
                  supervision (Any): Supervision to record.
        """
    def close(self) -> None:
        """Closes the Recorder and send the data if train_config is set."""
