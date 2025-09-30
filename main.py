import cv2
import open3d as o3d
from pypylon import pylon
from pathlib import Path

from src.stereo_inference import StereoInferenceOnnx
from src.stereo_calibrate import (
    read_stereo_calibration,
    rectify_image,
    read_camera_intrinsics,
)
from src.utils import (
    StereoImages,
    images_to_tensors,
    visualize_disparity,
    postprocess_disparity,
    create_point_cloud,
)

"""
Steps:
1. Load stereo calibration parameters
2. Load camera intrinsics, distance between cameras
3. Load stereo model
4. Connect to cameras
5. Grab images from both cameras
6. Rectify images
7. Images to tensors
8. Model inference
9. Show disparity map
10. Save disparity map, point cloud
"""


def main():
    print('Loading stereo calibration parameters and model...')
    # Load stereo calibration parameters
    stereo_map_left, stereo_map_right = read_stereo_calibration(
        'camera_configs/stereo_calibration.xml'
    )
    # Read camera intrinsics
    K, distance_between_cameras = read_camera_intrinsics(
        'camera_configs/left_camera_intrinsics.xml'
    )
    # Load stereo model
    stereo_model = StereoInferenceOnnx('models/fs_800_640.onnx')
    print('Parameters and stereo model loaded successfully.')

    # Load stereo image preprocessor
    processor = StereoImages(stereo_map_left, stereo_map_right)

    # Get all available cameras
    available_cameras = pylon.TlFactory.GetInstance().EnumerateDevices()
    if len(available_cameras) < 2:
        print(
            f'Error: Found only {len(available_cameras)} camera(s). Need at least 2 cameras.'
        )
        return

    # Connect to the first two available cameras
    right_camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateDevice(available_cameras[1])
    )
    left_camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateDevice(available_cameras[0])
    )

    # Open cameras
    left_camera.Open()
    right_camera.Open()

    exposure_time = 35000
    left_camera.ExposureTime.SetValue(exposure_time)
    right_camera.ExposureTime.SetValue(exposure_time)

    camera_width = 3200
    left_camera.Width.SetValue(camera_width)
    right_camera.Width.SetValue(camera_width)

    camera_height = 4000
    left_camera.Height.SetValue(camera_height)
    right_camera.Height.SetValue(camera_height)

    offset_x = 652
    left_camera.OffsetX.SetValue(offset_x)
    right_camera.OffsetX.SetValue(offset_x)

    offset_y = 52
    left_camera.OffsetY.SetValue(offset_y)
    right_camera.OffsetY.SetValue(offset_y)

    # Start grabbing continuously from both cameras
    left_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    right_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Create converters for both cameras
    left_converter = pylon.ImageFormatConverter()
    right_converter = pylon.ImageFormatConverter()

    # Configure converters to OpenCV BGR format
    left_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    left_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    right_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    right_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    frame_count = 0

    print("Press 'c' to capture frames from both cameras")
    print("Press 'ESC' to exit")

    while left_camera.IsGrabbing() and right_camera.IsGrabbing():
        # Retrieve results from both cameras with timeout
        left_result = left_camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException
        )
        right_result = right_camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException
        )

        if left_result.GrabSucceeded() and right_result.GrabSucceeded():
            # Convert the grabbed images to OpenCV format
            left_image = left_converter.Convert(left_result)
            right_image = right_converter.Convert(right_result)

            # Get the array from the image
            left_img = left_image.GetArray()
            right_img = right_image.GetArray()

            # Display the images
            cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
            cv2.imshow('Left Camera', left_img)
            cv2.imshow('Right Camera', right_img)

            # Check for key press
            k = cv2.waitKey(1)

            # 'c' key for capture
            if k == ord('c'):
                frame_count += 1
                print(f'Captured frame {frame_count}')
                
                proc_result = processor.process(
                    left_img, right_img,
                    stereo_model.input_width,
                    stereo_model.input_height
                )
                
                left_tensor, right_tensor = proc_result['tensors']
                scale = proc_result['scale']
                left_rectified = proc_result['processed_images'][0]

                # Stereo inference
                disparity_map = stereo_model(left_tensor, right_tensor)

                # Squeeze form 4D to 2D
                disparity_map = disparity_map.squeeze(0).squeeze(0)

                # Visualize disparity map
                disparity_map_viz = visualize_disparity(disparity_map)

                # Postprocess disparity map
                disparity_map_proc = postprocess_disparity(disparity_map)

                # Show disparity map
                cv2.imshow(
                    'Disparity Map', cv2.cvtColor(disparity_map_viz, cv2.COLOR_RGB2BGR)
                )

                # save disparity map
                output_path = Path('output/disp/')
                output_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(output_path / f'disp_{frame_count}.png'),
                    cv2.cvtColor(disparity_map_viz, cv2.COLOR_RGB2BGR),
                )

                # Create point cloud
                pcd = create_point_cloud(
                    disparity_map_proc,
                    left_rectified,
                    K,
                    distance_between_cameras,
                    scale=scale,
                )
                # Save point cloud
                output_path = Path('output/pcd/')
                output_path.mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud(
                    str(output_path / f'pcd_{frame_count}.ply'), pcd
                )

            # ESC key to exit
            elif k == 27:
                break

        # Release the grab results
        left_result.Release()
        right_result.Release()

    # Stop grabbing and close cameras
    left_camera.StopGrabbing()
    right_camera.StopGrabbing()
    left_camera.Close()
    right_camera.Close()


if __name__ == '__main__':
    main()
