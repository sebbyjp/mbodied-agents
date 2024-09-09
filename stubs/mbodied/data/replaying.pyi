import mbodied.types.sample
import types
import typing
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import MbImage as MbImage
from typing import Any

class Replayer:
    def __init__(self, path: str, file_keys=..., image_keys_to_save=...) -> None:
        """Initialize the Replayer class with the given parameters.

                Args:
                    path (str): Path to the HDF5 file.
                    file_keys (List[str], optional): List of keys in the file. Defaults to None.
                    image_keys_to_save (List[str], optional): List of image keys to save. Defaults to None.
        """
    def get_frames_path(self) -> types.str | None:
        """Get the path to the frames directory."""
    def recursive_do(self, do: typing.Callable, key: str = ..., prefix: str = ..., **kwargs) -> Any:
        """Recursively perform a function on each key in the HDF5 file.

                Args:
                    do (Callable): Function to perform.
                    key (str, optional): Key in the HDF5 file. Defaults to ''.
                    prefix (str, optional): Prefix for the key. Defaults to ''.
                    **kwargs: Additional arguments to pass to the function.

                Returns:
                    Any: Result of the function.
        """
    def get_unique_items(self, key: str) -> list:
        """Get unique items for a given key.

                Args:
                    key (str): Key in the HDF5 file.

                Returns:
                    List[str]: List of unique items.
        """
    def read_sample(self, index: int) -> tuple:
        """Read a sample from the HDF5 file at a given index.

                Args:
                    index (int): Index of the sample.

                Returns:
                    Tuple[dict, ...]: Tuple of dictionaries containing the sample data.
        """
    def get_structure(self, key: str = ..., prefix: str = ...) -> dict:
        """Get the structure of the HDF5 file.

                Args:
                    key (str, optional): Key in the HDF5 file. Defaults to ''.
                    prefix (str, optional): Prefix for the key. Defaults to ''.

                Returns:
                    dict: Structure of the HDF5 file.
        """
    def pack_one(self, index: int) -> mbodied.types.sample.Sample:
        """Pack a single sample into a Sample object.

                Args:
                    index (int): Index of the sample.

                Returns:
                    Sample: Sample object.
        """
    def pack(self) -> mbodied.types.sample.Sample:
        """Pack all samples into a Sample object with attributes being lists of samples.

                Returns:
                    Sample: Sample object containing all samples.
        """
    def sample(self, index=..., n=...) -> mbodied.types.sample.Sample:
        """Get a sample from the HDF5 file.

                Args:
                    index (Optional[Union[int, slice]], optional): Index or slice of the sample. Defaults to None.
                    n (int, optional): Number of samples to get. Defaults to 1.

                Returns:
                    Sample: Sample object.
        """
    def get_stats(self, key: str = ..., prefix: str = ...) -> dict:
        """Get statistics for a given key in the HDF5 file.

                Args:
                    key (str, optional): Key in the HDF5 file. Defaults to ''.
                    prefix (str, optional): Prefix for the key. Defaults to ''.

                Returns:
                    dict: Statistics for the given key.
        """
    def __iter__(self):
        """Iterate over the HDF5 file."""
    def __next__(self) -> tuple:
        """Get the next sample from the HDF5 file.

                Returns:
                    Tuple[dict, ...]: Tuple of dictionaries containing the sample data.
        """
    def close(self) -> None:
        """Close the HDF5 file."""
def clean_folder(folder: str, image_keys_to_save: list) -> None:
    """Clean the folder by iterating through the files and asking for deletion.

        Args:
            folder (str): Path to the folder.
            image_keys_to_save (List[str]): List of image keys to save.
    """

class FolderReplayer:
    def __init__(self, path: str) -> None:
        """Initialize the FolderReplayer class with the given path.

                Args:
                    path (str): Path to the folder containing HDF5 files.
        """
    def __iter__(self):
        """Iterate through the HDF5 files in the folder."""
def to_dataset(folder: str, name: str, description=..., **kwargs) -> None:
    """Convert the folder of HDF5 files to a Hugging Face dataset.

        Args:
            folder (str): Path to the folder containing HDF5 files.
            name (str): Name of the dataset.
            description (str, optional): Description of the dataset. Defaults to None.
            **kwargs: Additional arguments to pass to the Dataset.push_to_hub method.
    """
def parse_slice(s: str) -> types.int | slice:
    '''Parse a string to an integer or slice.

        Args:
            s (str): String to parse.

        Returns:
            Union[int, slice]: Integer or slice.

        Examples:
            >>> lst = [0, 1, 2, 3, 4, 5]
            >>> lst[parse_slice("1")]
            1
            >>> lst[parse_slice("1:5:2")]
            [1, 3]
    '''
