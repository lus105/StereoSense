from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class StereoCalibration:
    def __init__(
        self,
        left_path: str,
        right_path: str,
        chessboard_height: int,
        chessboard_width: int,
        square_size: float,
        distance_between_cameras: float = 0.1,
        file_extension: str = '.png',
        output_path: str = 'output/',
    ) -> None:
        """Initialize stereo calibration.

        Args:
            left_path (str): path to left images
            right_path (str): path to right images
            chessboard_height (int): height of chessboard
            chessboard_width (int): width of chessboard
            square_size (float): size of square in meters
            distance_between_cameras (float, optional): distance between cameras
                in meters. Defaults to 0.1.
            file_extension (str): file extension of images
            output_path (str): path to output directory
        """
        self._left_path = Path(left_path)
        self._right_path = Path(right_path)
        self._chessboard_size = (chessboard_width, chessboard_height)
        self._square_size = square_size
        self._distance_between_cameras = distance_between_cameras
        self._file_extension = file_extension
        self._output_path = Path(output_path)

        self._chessboard_3d_left_points: list[np.ndarray] = []
        self._chessboard_2d_left_points: list[np.ndarray] = []
        self._chessboard_3d_right_points: list[np.ndarray] = []
        self._chessboard_2d_right_points: list[np.ndarray] = []

        self._calibrated_mtx_left: np.ndarray = None
        self._calibrated_mtx_right: np.ndarray = None

        self._stereo_map_left: tuple[np.ndarray, np.ndarray] = []
        self._stereo_map_right: tuple[np.ndarray, np.ndarray] = []

        self._left_image_size: tuple[int, int] = None
        self._right_image_size: tuple[int, int] = None

        self._gather_image_paths()

    def _gather_image_paths(self) -> None:
        """Initialize paths to left and right images.

        The left and right images are stored in separate directories as follows:
        - images
            - left
                - left_0000.png
                - left_0001.png
                - ...
            - right
                - right_0000.png
                - right_0001.png
                - ...

        """
        self._left_image_paths = sorted(
            [p for p in self._left_path.iterdir() if p.suffix == self._file_extension]
        )

        self._right_image_paths = sorted(
            [p for p in self._right_path.iterdir() if p.suffix == self._file_extension]
        )

    def _create_3d_chessboard_points(self) -> np.ndarray:
        """Create 3D chessboard points for a given chessboard size and square size.


        Returns:
            np.ndarray: array of 3D points: N = number of squares in chessboard
        """

        # create Nx3 array of 3D points: N = number of squares in chessboard
        chessboard_points = np.zeros((np.prod(self._chessboard_size), 3), np.float32)
        # fill x, y coordinates with indices of chessboard points
        chessboard_points[:, :2] = np.indices(self._chessboard_size).T.reshape(-1, 2)
        # scale coordinates by square size
        chessboard_points *= self._square_size
        return chessboard_points

    def _create_2d_chessboard_points(self, image: np.ndarray) -> np.ndarray:
        """Create 2D chessboard points for a given image and chessboard size.

        Args:
            image (np.ndarray): image of chessboard

        Returns:
            np.ndarray: Nx1x2 array of 2D points: N = number of corners found in the image
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # find chessboard corners
        found, corners = cv2.findChessboardCorners(image, self._chessboard_size, None)
        if not found:
            return None
        # refine corner positions
        corners = cv2.cornerSubPix(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )
        return corners

    def _create_chessboard_points(self, save_images: bool = False) -> None:
        """Create 3D and 2D chessboard points for left and right images.

        Args:
            save_images (bool, optional): Save images with points. Defaults to False.
        """
        (self._output_path / 'left').mkdir(exist_ok=True, parents=True)
        (self._output_path / 'right').mkdir(exist_ok=True, parents=True)

        for left_image_path, right_image_path in tqdm(
            zip(self._left_image_paths, self._right_image_paths)
        ):
            left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
            right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
            left_corners = self._create_2d_chessboard_points(left_image)
            right_corners = self._create_2d_chessboard_points(right_image)

            if save_images:
                left_image = cv2.drawChessboardCorners(
                    left_image, self._chessboard_size, left_corners, True
                )
                right_image = cv2.drawChessboardCorners(
                    right_image, self._chessboard_size, right_corners, True
                )
                left_image_name = left_image_path.name
                right_image_name = right_image_path.name
                left_image_path = self._output_path / 'left' / left_image_name
                right_image_path = self._output_path / 'right' / right_image_name

                cv2.imwrite(str(left_image_path), left_image)
                cv2.imwrite(str(right_image_path), right_image)

            if left_corners is not None and right_corners is not None:
                self._chessboard_3d_left_points.append(
                    self._create_3d_chessboard_points()
                )
                self._chessboard_2d_left_points.append(left_corners)
                self._chessboard_3d_right_points.append(
                    self._create_3d_chessboard_points()
                )
                self._chessboard_2d_right_points.append(right_corners)

        self._left_image_size = (left_image.shape[1], left_image.shape[0])
        self._right_image_size = (right_image.shape[1], right_image.shape[0])

    def calibrate_single_camera(
        self, left: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calibrate single camera.

        Args:
            left (bool): True for left camera, False for right camera

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: camera matrix,
              distortion coefficients, new camera matrix, ROI
        """
        # left camera calibration
        print('Calibrating left camera...')
        if left:
            ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                self._chessboard_3d_left_points,
                self._chessboard_2d_left_points,
                (self._left_image_size),
                None,
                None,
            )

            newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(
                mtx_left, dist_left, self._left_image_size, 1, self._left_image_size
            )
            print('Left camera calibration complete!')

            return mtx_left, dist_left, newcameramtx_left, roi_left
        else:
            ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = (
                cv2.calibrateCamera(
                    self._chessboard_3d_right_points,
                    self._chessboard_2d_right_points,
                    (self._right_image_size),
                    None,
                    None,
                )
            )
            print('Getting optimal new camera matrix for right camera...')
            newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(
                mtx_right, dist_right, self._right_image_size, 1, self._right_image_size
            )
            print('Right camera calibration complete!')

            return mtx_right, dist_right, newcameramtx_right, roi_right

    def calibrate(self) -> tuple[np.ndarray, np.ndarray]:
        """Calibrate stereo camera.

        The calibration process consists of the following steps:
        1. Calibrate left camera
        2. Calibrate right camera
        3. Stereo calibration
        4. Stereo rectification
        5. Stereo mapping


        Returns:
            tuple[np.ndarray, np.ndarray]: stereo_map_left, stereo_map_right
        """
        self._create_chessboard_points(save_images=True)

        # left camera calibration
        mtx_left, dist_left, newcameramtx_left, roi_left = self.calibrate_single_camera(
            left=True
        )
        print('Left camera matrix', newcameramtx_left)
        self._calibrated_mtx_left = newcameramtx_left

        # right camera calibration
        mtx_right, dist_right, newcameramtx_right, roi_right = (
            self.calibrate_single_camera(left=False)
        )
        print('Right camera matrix', newcameramtx_right)
        self._calibrated_mtx_right = newcameramtx_right

        # stereo calibration
        print('Calibrating stereo...')
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        (
            ret_stereo,
            newcameramtx_left,
            dist_left,
            newcameramtx_right,
            dist_right,
            rot,
            trans,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            self._chessboard_3d_left_points,
            self._chessboard_2d_left_points,
            self._chessboard_2d_right_points,
            newcameramtx_left,
            dist_left,
            newcameramtx_right,
            dist_right,
            self._left_image_size,
            criteria=criteria,
            flags=flags,
        )
        print('translation vector', trans)
        print('rotation matrix', rot)

        # stereo rectification
        print('Rectifying stereo camera...')
        rect_left, rect_right, proj_left, proj_right, Q, roi_left, roi_right = (
            cv2.stereoRectify(
                newcameramtx_left,
                dist_left,
                newcameramtx_right,
                dist_right,
                self._left_image_size,
                rot,
                trans,
                1,
                (0, 0),
            )
        )

        # stereo mapping
        print('Mapping left and right cameras...')
        stereo_map_left = cv2.initUndistortRectifyMap(
            newcameramtx_left,
            dist_left,
            rect_left,
            proj_left,
            self._left_image_size,
            cv2.CV_16SC2,
        )
        stereo_map_right = cv2.initUndistortRectifyMap(
            newcameramtx_right,
            dist_right,
            rect_right,
            proj_right,
            self._right_image_size,
            cv2.CV_16SC2,
        )
        print('Calibration complete!')
        self._stereo_map_left = stereo_map_left
        self._stereo_map_right = stereo_map_right
        return stereo_map_left, stereo_map_right

    def save_stereo_calibration(self) -> None:
        """Save stereo calibration to XML file."""

        print('Saving stereo calibration...')
        # create output directory if it doesn't exist
        self._output_path.mkdir(exist_ok=True, parents=True)
        xml_path = self._output_path / 'stereo_calibration.xml'
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
        fs.write('stereo_map_left_x', self._stereo_map_left[0])
        fs.write('stereo_map_left_y', self._stereo_map_left[1])
        fs.write('stereo_map_right_x', self._stereo_map_right[0])
        fs.write('stereo_map_right_y', self._stereo_map_right[1])
        fs.release()

    def save_intrinsics(self, left: bool = True) -> None:
        """Save camera intrinsics to XML file.

        Args:
            left (bool, optional): True for left camera, False for right camera. Defaults to True.
        """
        print('Saving camera intrinsics...')
        # create output directory if it doesn't exist
        self._output_path.mkdir(exist_ok=True, parents=True)
        if left:
            cam = 'left'
        else:
            cam = 'right'

        xml_path = self._output_path / f'{cam}_camera_intrinsics.xml'
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
        if left:
            fs.write('camera_matrix', self._calibrated_mtx_left)
        else:
            fs.write('camera_matrix', self._calibrated_mtx_right)

        fs.write('distance_between_cameras', self._distance_between_cameras)
        fs.release()


def read_stereo_calibration(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read stereo calibration from XML file.

    Args:
        xml_path (str): path to XML file

    Returns:
        tuple[np.ndarray, np.ndarray]: stereo_map_left, stereo_map_right
    """

    # use cv2.FileStorage to read stereo calibration
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    stereo_map_left_x = fs.getNode('stereo_map_left_x').mat()
    stereo_map_left_y = fs.getNode('stereo_map_left_y').mat()
    stereo_map_right_x = fs.getNode('stereo_map_right_x').mat()
    stereo_map_right_y = fs.getNode('stereo_map_right_y').mat()
    fs.release()
    return (stereo_map_left_x, stereo_map_left_y), (
        stereo_map_right_x,
        stereo_map_right_y,
    )


def read_camera_intrinsics(xml_path: str) -> tuple[np.ndarray, float]:
    """Read stereo calibration from XML file.

    Args:
        xml_path (str): path to XML file

    Returns:
        tuple[np.ndarray, float]: camera matrix, distance between cameras
    """

    # use cv2.FileStorage to read stereo calibration
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    distance_between_cameras = fs.getNode('distance_between_cameras').real()
    fs.release()
    return camera_matrix, distance_between_cameras


def rectify_images(
    left_image: np.ndarray,
    right_image: np.ndarray,
    stereo_map_left: tuple[np.ndarray, np.ndarray],
    stereo_map_right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify left and right images using stereo mapping.

    Args:
        left_image (np.ndarray): left_image
        right_image (np.ndarray): right_image
        stereo_map_left (tuple[np.ndarray, np.ndarray]): stereo_map_left
        stereo_map_right (tuple[np.ndarray, np.ndarray]): stereo_map_right

    Returns:
        tuple[np.ndarray, np.ndarray]: left_rectified, right_rectified (images)
    """
    # rectify images
    left_rectified = cv2.remap(
        left_image,
        stereo_map_left[0],
        stereo_map_left[1],
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0,
    )
    right_rectified = cv2.remap(
        right_image,
        stereo_map_right[0],
        stereo_map_right[1],
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0,
    )
    return left_rectified, right_rectified
