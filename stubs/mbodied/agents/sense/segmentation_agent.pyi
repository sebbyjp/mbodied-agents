import numpy as np
from _typeshed import Incomplete
from mbodied.agents.sense.sensory_agent import SensoryAgent as SensoryAgent
from mbodied.types.sense.scene import BBox2D as BBox2D, PixelCoords as PixelCoords
from mbodied.types.sense.vision import Image as Image

class SegmentationAgent(SensoryAgent):
    """An image segmentation agent that uses a remote segmentation server to segment objects in an image."""
    def __init__(self, model_src: str = 'https://api.mbodi.ai/sense/', model_kwargs: Incomplete | None = None, **kwargs) -> None: ...
    def act(self, image: Image, input_data: BBox2D | list[BBox2D] | PixelCoords, *args, api_name: str = '/segment', **kwargs) -> tuple[Image, np.ndarray]:
        """Perform image segmentation using the remote segmentation server.

        Args:
            image (Image): The image to act on.
            input_data (Union[BBox2D, List[BBox2D], PixelCoords]): The input data for segmentation, either a bounding box,
                a list of bounding boxes, or pixel coordinates.
            *args: Variable length argument list.
            api_name (str): The name of the API endpoint to use.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple[Image, np.ndarray]: The segmented image and the masks of the segmented objects.
        """
