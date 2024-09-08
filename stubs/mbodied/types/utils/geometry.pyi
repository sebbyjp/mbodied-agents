import numpy as np
from shapely.geometry import Polygon as Polygon

def rotation_matrix(deg: float) -> np.ndarray:
    """Generate a 2x2 rotation matrix for a given angle in degrees."""
def rotation_to_transformation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a transformation matrix."""
def pose_to_transformation_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Convert a pose (position and rotation) to a transformation matrix."""
def transformation_matrix_to_pose(T: np.ndarray) -> tuple:
    """Extract position and rotation matrix from a transformation matrix."""
def transformation_matrix_to_position(T: np.ndarray) -> np.ndarray:
    """Extract position from a transformation matrix."""
def transformation_matrix_to_rotation(T: np.ndarray) -> np.ndarray:
    """Extract rotation matrix from a transformation matrix."""
def rpy_to_rotation_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw angles (in radians) to a rotation matrix."""
def rotation_matrix_to_rpy(R: np.ndarray, unit: str = 'rad') -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw angles (in radians or degrees)."""
def rotation_matrix_to_angular_velocity(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an angular velocity vector."""
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion."""
def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
    """Generate a skew-symmetric matrix from a vector."""
def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Compute the rotation matrix from an angular velocity vector."""
def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector of the input vector."""
def get_rotation_matrix_from_two_points(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """Generate a rotation matrix that aligns one point with another."""
def trim_scale(x: np.ndarray, threshold: float) -> np.ndarray:
    """Scale down the input array if its maximum absolute value exceeds the threshold."""
def soft_squash(x: np.ndarray, x_min: float = -1, x_max: float = 1, margin: float = 0.1) -> np.ndarray:
    """Softly squash the values of an array within a specified range with margins."""
def soft_squash_multidim(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Apply soft squashing to a multi-dimensional array."""
def squared_exponential_kernel(X1: np.ndarray, X2: np.ndarray, hyp: dict) -> np.ndarray:
    """Compute the squared exponential (SE) kernel between two sets of points."""
def leveraged_squared_exponential_kernel(X1: np.ndarray, X2: np.ndarray, L1: np.ndarray, L2: np.ndarray, hyp: dict) -> np.ndarray:
    """Compute the leveraged SE kernel between two sets of points."""
def is_point_in_polygon(point: np.ndarray, polygon: Polygon) -> bool:
    """Check if a point is inside a given polygon."""
def is_point_feasible(point: np.ndarray, obstacles: list) -> bool:
    """Check if a point is feasible (not inside any obstacles)."""
def is_line_connectable(p1: np.ndarray, p2: np.ndarray, obstacles: list) -> bool:
    """Check if a line between two points is connectable (does not intersect any obstacles)."""
def interpolate_constant_velocity_trajectory(traj_anchor: np.ndarray, velocity: float = 1.0, hz: int = 100, order: int = ...) -> tuple:
    """Interpolate a trajectory to achieve constant velocity."""
def depth_image_to_pointcloud(depth_img: np.ndarray, cam_matrix: np.ndarray) -> np.ndarray:
    """Convert a scaled depth image to a point cloud."""
def compute_view_params(camera_pos: np.ndarray, target_pos: np.ndarray, up_vector: np.ndarray = ...) -> tuple:
    """Compute view parameters (azimuth, distance, elevation, lookat) for a camera."""
def sample_points_in_3d(n_sample: int, x_range: list, y_range: list, z_range: list, min_dist: float, xy_margin: float = 0.0) -> np.ndarray:
    """Sample points in 3D space ensuring a minimum distance between them."""
def quintic_trajectory(start_pos: np.ndarray, start_vel: np.ndarray, start_acc: np.ndarray, end_pos: np.ndarray, end_vel: np.ndarray, end_acc: np.ndarray, duration: float, num_points: int, max_velocity: float, max_acceleration: float) -> tuple:
    """Generate a quintic trajectory with velocity and acceleration constraints."""
def passthrough_filter(pcd: np.ndarray, axis: int, interval: list) -> np.ndarray:
    """Filter a point cloud along a specified axis within a given interval."""
def remove_duplicates(pointcloud: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Remove duplicate points from a point cloud within a given threshold."""
def remove_duplicates_with_reference(pointcloud: np.ndarray, reference_point: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Remove duplicate points close to a specific reference point within a given threshold."""
def downsample_pointcloud(pointcloud: np.ndarray, grid_size: float) -> np.ndarray:
    """Downsample a point cloud based on a specified grid size."""
