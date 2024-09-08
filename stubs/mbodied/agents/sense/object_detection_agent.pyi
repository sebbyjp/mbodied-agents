from _typeshed import Incomplete
from mbodied.agents.sense.sensory_agent import SensoryAgent as SensoryAgent
from mbodied.types.sense.scene import Scene as Scene
from mbodied.types.sense.vision import Image as Image

class ObjectDetectionAgent(SensoryAgent):
    """A object detection agent that uses a remote object detection, i.e. YOLO server, to detect objects in an image."""
    def __init__(self, model_src: str = 'https://api.mbodi.ai/sense/', model_kwargs: Incomplete | None = None, **kwargs) -> None: ...
    def act(self, image: Image, objects: list[str] | str, *args, api_name: str = '/detect', **kwargs) -> Scene:
        """Act based on the prompt and image using the remote object detection server.

        Args:
            image (Image): The image to act on.
            objects (list[str] | str): The objects to detect in the image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Scene: The scene data with the detected objects.
        """
