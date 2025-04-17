import cv2
import numpy as np
import open3d as o3d
from typing import Optional

def images_to_tensors(
        left_img: np.ndarray,
        right_img: np.ndarray,
        target_height: int = 800,
        target_width: int = 640
        ) -> tuple[np.ndarray, np.ndarray]:
    """    Preprocess images for ONNX inference:
    1. Resize if target dimensions are provided
    2. Convert to RGB if needed
    3. Normalize to float32 [0-255]
    4. Transpose from HWC to CHW format
    5. Add batch dimension

    Args:
        left_img (np.ndarray): left image.
        right_img (np.ndarray): right image.
        target_height (int, optional): target height. Defaults to 800.
        target_width (int, optional): target width. Defaults to 640.

    Returns:
        tuple[np.ndarray, np.ndarray]: left and right image tensors.
    """
    # Calculate image scale
    scale = min(target_height / left_img.shape[0], target_width / left_img.shape[1])

    # Resize if dimensions are provided
    if target_height is not None and target_width is not None:
        left_img = cv2.resize(left_img, (target_width, target_height))
        right_img = cv2.resize(right_img, (target_width, target_height))

    # Convert to RGB if in BGR format
    if left_img.shape[2] == 3:  # If color image
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # Convert to float32 [0-255]
    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)

    # Transpose from HWC to CHW format
    left_img = left_img.transpose(2, 0, 1)  # (C, H, W)
    right_img = right_img.transpose(2, 0, 1)  # (C, H, W)

    # Add batch dimension
    left_img = np.expand_dims(left_img, axis=0)  # (1, C, H, W)
    right_img = np.expand_dims(right_img, axis=0)  # (1, C, H, W)

    return left_img, right_img, scale


def visualize_disparity(
        disparity: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        invalid_thres: float = np.inf,
        color_map: int = cv2.COLORMAP_TURBO,
        cmap = None,
        other_output: dict[str, any] = {}
    ) -> np.ndarray:
    """Colorize and visualize a disparity map with options for custom color mapping.

    Args:
        disparity (np.ndarray): Input disparity map as a 2D array.
        min_val (Optional[float], optional): Minimum disparity value for normalization.
            If None, it's calculated from valid areas. Defaults to None.
        max_val (Optional[float], optional): Maximum disparity value for normalization.
            If None, it's calculated from valid areas. Defaults to None.
        invalid_thres (float, optional): Threshold above which disparity values
            are considered invalid. Defaults to np.inf.
        color_map (int, optional): OpenCV colormap ID to apply to the disparity.
            Defaults to cv2.COLORMAP_TURBO.
        cmap (Optional[Colormap], optional): Matplotlib colormap object to use instead of OpenCV's.
            Defaults to None.
        other_output (Dict[str, Any], optional): Dictionary to store the computed
            min and max values. Defaults to {}.

    Returns:
        np.ndarray: Colorized disparity map as an RGB image (H×W×3 uint8 array).
    """
    disparity_cp = disparity.copy()

    H, W = disparity_cp.shape[:2]

    invalid_mask = disparity_cp >= invalid_thres
    if (invalid_mask==0).sum()==0:
        other_output['min_val'] = None
        other_output['max_val'] = None
        return np.zeros((H, W, 3))
    
    if min_val is None:
        min_val = disparity_cp[invalid_mask==0].min()

    if max_val is None:
        max_val = disparity_cp[invalid_mask==0].max()

    other_output['min_val'] = min_val
    other_output['max_val'] = max_val

    vis = ((disparity_cp-min_val)/(max_val-min_val)).clip(0, 1) * 255

    if cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
    else:
        vis = cmap(vis.astype(np.uint8))[...,:3]*255

    if invalid_mask.any():
        vis[invalid_mask] = 0

    return vis.astype(np.uint8)

def postprocess_disparity(
        disparity: np.ndarray,
    ) -> np.ndarray:
    """Remove non-overlapping observations between left and right images from
    point cloud, so the remaining points are more reliable.

    Args:
        disparity (np.ndarray): Disparity map as a 2D array.

    Returns:
        np.ndarray: Postprocessed disparity map.
    """
    yy, xx = np.meshgrid(np.arange(disparity.shape[0]), np.arange(disparity.shape[1]), indexing='ij')
    us_right = xx - disparity
    invalid = us_right < 0
    disparity[invalid] = np.inf
    return disparity

def depth_to_xyzmap(depth: np.ndarray,
                 K: np.array,
                 uvs: np.ndarray=None,
                 zmin: float = 0.1
    ) -> np.ndarray:
    """Convert a depth map to a 3D point cloud in camera coordinates.

    Args:
        depth (np.ndarray): Depth map as a 2D array where values represent Z-distance from camera.
        K (np.array): Camera intrinsic matrix (3x3) containing focal lengths and principal point.
        uvs (np.ndarray, optional): Specific pixel coordinates as (u,v) pairs to process. If None,
            all pixels are processed. Defaults to None.
        zmin (float, optional): Minimum valid depth value. Pixels with depth < zmin are treated as
            invalid. Defaults to 0.1.

    Returns:
        np.ndarray: XYZ coordinate map as an (H,W,3) array, where each pixel contains
            its 3D coordinates (x,y,z) in camera space.
    """
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]

    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H),np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]

    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map

def points_to_pcd(
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
    """Convert numpy arrays to an Open3D point cloud.

    Args:
        points (np.ndarray): Point coordinates as an Nx3 array of (x,y,z) positions.
        colors (Optional[np.ndarray], optional): RGB colors for each point as an Nx3 array
            with values in [0,1] or [0,255]. Defaults to None.
        normals (Optional[np.ndarray], optional): Surface normals for each point
            as an Nx3 array. Defaults to None.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object with the specified points,
            colors, and normals.
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def create_point_cloud(
        disparity: np.ndarray,
        image: np.ndarray,
        K: np.ndarray = None,
        dist_between_cameras: float = 0.1,
        scale: float = 1.0,
        z_far: int = 10
    ) -> o3d.geometry.PointCloud:

    K[:2] *= scale
    depth = K[0,0] * dist_between_cameras / disparity
    xyz_map = depth_to_xyzmap(depth, K)

    pcd = points_to_pcd(xyz_map.reshape(-1, 3), image.reshape(-1, 3))
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)

    return pcd