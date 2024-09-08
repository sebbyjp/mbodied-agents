from PIL import Image

def draw_bounding_box_with_label(image_path: str, bbox: tuple, label: str, color: str = 'red', width: int = 2) -> Image:
    '''Draws a bounding box with a label and shading on an image.

    Args:
        image_path (str): Path to the image file.
        bbox (tuple): Bounding box coordinates as (left, top, right, bottom).
        label (str): Label text for the bounding box.
        color (str): Color of the bounding box and label text. Default is red.
        width (int): Width of the bounding box lines. Default is 2.

    Returns:
        Image: Image object with the bounding box and label drawn.

    Example:
        image_with_bbox = draw_bounding_box_with_label("path/to/image.jpg", (50, 50, 150, 150), "Object")
    '''
