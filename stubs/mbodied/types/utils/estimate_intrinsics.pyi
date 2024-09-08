import numpy as np

def estimate_intrinsic_parameters(unscaled_depth_map: np.ndarray, image: np.ndarray, seg_map: np.ndarray = None) -> dict:
    """Estimate intrinsic camera parameters given an unscaled depth map, image, and optionally a semantic segmentation map.

    Args:
        unscaled_depth_map (np.ndarray): Unscaled depth map.
        image (np.ndarray): Image corresponding to the depth map.
        seg_map (np.ndarray, optional): Semantic segmentation map. Defaults to None.

    Returns:
        dict: Estimated intrinsic parameters including focal lengths and principal point.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from mbodied.agents.sense.utils.estimate_intrinsics import estimate_intrinsic_parameters
        >>> unscaled_depth_map = np.random.rand(480, 640)
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> estimate_intrinsic_parameters(unscaled_depth_map, image)
        {'fx': 1.0, 'fy': 1.0, 'cx': 320.0, 'cy': 240.0}
    """
