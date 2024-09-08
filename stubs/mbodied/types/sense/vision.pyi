import numpy as np
from PIL.Image import Image as PILImage
from _typeshed import Incomplete
from datasets.features import Features as Features
from gymnasium import spaces
from mbodied.types.ndarray import NumpyArray as NumpyArray
from mbodied.types.sample import Sample as Sample
from pydantic import AnyUrl, Base64Str, ConfigDict, FilePath, InstanceOf as InstanceOf
from typing import Any
from typing_extensions import Literal

SupportsImage = np.ndarray | PILImage | Base64Str | AnyUrl | FilePath

class Image(Sample):
    '''An image sample that can be represented in various formats.

    The image can be represented as a NumPy array, a base64 encoded string, a file path, a PIL Image object,
    or a URL. The image can be resized to and from any size and converted to and from any supported format.

    Attributes:
        array (Optional[np.ndarray]): The image represented as a NumPy array.
        base64 (Optional[Base64Str]): The base64 encoded string of the image.
        path (Optional[FilePath]): The file path of the image.
        pil (Optional[PILImage]): The image represented as a PIL Image object.
        url (Optional[AnyUrl]): The URL of the image.
        size (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.
        encoding (Optional[Literal["png", "jpeg", "jpg", "bmp", "gif"]]): The encoding of the image.

    Example:
        >>> image = Image("https://example.com/image.jpg")
        >>> image = Image("/path/to/image.jpg")
        >>> image = Image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4Q3zaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA")

        >>> jpeg_from_png = Image("path/to/image.png", encoding="jpeg")
        >>> resized_image = Image(image, size=(224, 224))
        >>> pil_image = Image(image).pil
        >>> array = Image(image).array
        >>> base64 = Image(image).base64
    '''
    model_config: ConfigDict
    array: NumpyArray
    size: tuple[int, int]
    pil: InstanceOf[PILImage] | None
    encoding: Literal['png', 'jpeg', 'jpg', 'bmp', 'gif']
    base64: InstanceOf[Base64Str] | None
    url: InstanceOf[AnyUrl] | str | None
    path: FilePath | None
    @classmethod
    def supports(cls, arg: SupportsImage) -> bool: ...
    def __init__(self, arg: SupportsImage = None, url: str | None = None, path: str | None = None, base64: str | None = None, array: np.ndarray | None = None, pil: PILImage | None = None, encoding: str | None = 'jpeg', size: tuple | None = None, bytes_obj: bytes | None = None, **kwargs) -> None:
        """Initializes an image. Either one source argument or size tuple must be provided.

        Args:
          arg (SupportsImage, optional): The primary image source.
          url (Optional[str], optional): The URL of the image.
          path (Optional[str], optional): The file path of the image.
          base64 (Optional[str], optional): The base64 encoded string of the image.
          array (Optional[np.ndarray], optional): The numpy array of the image.
          pil (Optional[PILImage], optional): The PIL image object.
          encoding (Optional[str], optional): The encoding format of the image. Defaults to 'jpeg'.
          size (Optional[Tuple[int, int]], optional): The size of the image as a (width, height) tuple.
          **kwargs: Additional keyword arguments.
        """
    @staticmethod
    def from_base64(base64_str: str, encoding: str, size: Incomplete | None = None) -> Image:
        """Decodes a base64 string to create an Image instance.

        Args:
            base64_str (str): The base64 string to decode.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
    @staticmethod
    def open(path: str, encoding: str = 'jpeg', size: Incomplete | None = None) -> Image:
        """Opens an image from a file path.

        Args:
            path (str): The path to the image file.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
    @staticmethod
    def pil_to_data(image: PILImage, encoding: str, size: Incomplete | None = None) -> dict:
        """Creates an Image instance from a PIL image.

        Args:
            image (PIL.Image.Image): The source PIL image from which to create the Image instance.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
    @staticmethod
    def load_url(url: str, download: bool = False) -> PILImage | None:
        """Downloads an image from a URL or decodes it from a base64 data URI.

        Args:
            url (str): The URL of the image to download, or a base64 data URI.

        Returns:
            PIL.Image.Image: The downloaded and decoded image as a PIL Image object.
        """
    @classmethod
    def from_bytes(cls, bytes_data: bytes, encoding: str = 'jpeg', size: Incomplete | None = None) -> Image:
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
    @staticmethod
    def bytes_to_data(bytes_data: bytes, encoding: str = 'jpeg', size: Incomplete | None = None) -> dict:
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
    @classmethod
    def validate_kwargs(cls, values) -> dict: ...
    def save(self, path: str, encoding: str | None = None, quality: int = 10) -> None:
        """Save the image to the specified path.

        If the image is a JPEG, the quality parameter can be used to set the quality of the saved image.
        The path attribute of the image is updated to the new file path.

        Args:
            path (str): The path to save the image to.
            encoding (Optional[str]): The encoding to use for saving the image.
            quality (int): The quality to use for saving the image.
        """
    def show(self) -> None: ...
    def space(self) -> spaces.Box:
        """Returns the space of the image."""
    def exclude_pil(self) -> dict:
        """Convert the image to a base64 encoded string."""
    def dump(self, *args, as_field: str | None = None, **kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
    def infer_features_dict(self) -> Features:
        """Infer features of the image."""
