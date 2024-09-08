import numpy as np
import pyrealsense2.pyrealsense2 as rs
from _typeshed import Incomplete

class RealsenseCamera:
    """A class to handle capturing images from an Intel RealSense camera and encoding camera intrinsics.

    Attributes:
        width (int): Width of the image frames.
        height (int): Height of the image frames.
        fps (int): Frames per second for the video stream.
        pipeline (rs.pipeline): RealSense pipeline for streaming.
        config (rs.config): Configuration for the RealSense pipeline.
        profile (rs.pipeline_profile): Pipeline profile containing stream settings.
        depth_sensor (rs.sensor): Depth sensor of the RealSense camera.
        depth_scale (float): Depth scale factor for the RealSense camera.
        align (rs.align): Object to align depth frames to color frames.
    """
    width: Incomplete
    height: Incomplete
    fps: Incomplete
    pipeline: Incomplete
    config: Incomplete
    profile: Incomplete
    depth_sensor: Incomplete
    depth_scale: Incomplete
    align: Incomplete
    rs: Incomplete
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        """Initialize the RealSense camera with the given dimensions and frame rate.

        Args:
            width (int): Width of the image frames.
            height (int): Height of the image frames.
            fps (int): Frames per second for the video stream.
        """
    def capture_realsense_images(self) -> tuple[np.ndarray, np.ndarray, 'rs.intrinsics', np.ndarray]:
        """Capture color and depth images from the RealSense camera along with intrinsics.

        Returns:
            tuple: color_image (np.ndarray), depth_image (np.ndarray),
                   intrinsics (rs.intrinsics), intrinsics_matrix (np.ndarray)
        """
    @staticmethod
    def serialize_intrinsics(intrinsics: rs.intrinsics) -> dict:
        """Serialize camera intrinsics to a dictionary.

        Args:
            intrinsics (rs.intrinsics): The intrinsics object to serialize.

        Returns:
            dict: Serialized intrinsics as a dictionary.
        """
    @staticmethod
    def intrinsics_to_base64(intrinsics: rs.intrinsics) -> str:
        """Convert camera intrinsics to a base64 string.

        Args:
            intrinsics (rs.intrinsics): The intrinsics object to encode.

        Returns:
            str: Base64 encoded string of the intrinsics.
        """
    @staticmethod
    def base64_to_intrinsics(base64_str: str) -> rs.intrinsics:
        """Convert a base64 encoded string to an rs.intrinsics object.

        Args:
            base64_str (str): Base64 encoded string representing camera intrinsics.

        Returns:
            rs.intrinsics: An rs.intrinsics object with the decoded intrinsics data.
        """
    @staticmethod
    def matrix_and_distortion_to_intrinsics(image_height: int, image_width: int, matrix: np.ndarray, coeffs: np.ndarray) -> rs.intrinsics:
        """Convert a 3x3 intrinsic matrix and a 1x5 distortion coefficients array to an rs.intrinsics object.

        Args:
            image_height (int): The height of the image.
            image_width (int): The width of the image.
            matrix (np.ndarray): A 3x3 intrinsic matrix.
            coeffs (np.ndarray): A 1x5 array of distortion coefficients.

        Returns:
            rs.intrinsics: An rs.intrinsics object with the given intrinsics data.

        Example:
            >>> matrix = np.array([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]])
            >>> coeffs = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
            >>> intrinsics = RealsenseCamera.matrix_and_distortion_to_intrinsics(480, 640, matrix, coeffs)
            >>> expected = rs.intrinsics()
            >>> expected.width = 640
            >>> expected.height = 480
            >>> expected.ppx = 319.5
            >>> expected.ppy = 239.5
            >>> expected.fx = 525.0
            >>> expected.fy = 525.0
            >>> expected.model = rs.distortion.none
            >>> expected.coeffs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
            >>> assert (
            ...     intrinsics.width == expected.width
            ...     and intrinsics.height == expected.height
            ...     and intrinsics.ppx == expected.ppx
            ...     and intrinsics.ppy == expected.ppy
            ...     and intrinsics.fx == expected.fx
            ...     and intrinsics.fy == expected.fy
            ...     and intrinsics.model == expected.model
            ...     and np.allclose(intrinsics.coeffs, expected.coeffs)
            ... )
        """
    @staticmethod
    def pixel_to_3dpoint_realsense(centroid: tuple, depth: float, realsense_intrinsics: rs.intrinsics) -> np.ndarray:
        """Convert a 2D pixel coordinate to a 3D point using the depth and camera intrinsics.

        Args:
            centroid (tuple): The (u, v) coordinates of the pixel.
            depth (float): The depth value at the pixel.
            realsense_intrinsics (rs.intrinsics): Camera intrinsics.

        Returns:
            np.ndarray: The 3D coordinates of the point.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> estimator.pixel_to_3dpoint_realsense((320, 240), 1.5, realsense_intrinsics)
        """
