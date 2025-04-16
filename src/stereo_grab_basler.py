import cv2
import os
from pypylon import pylon
import time

def main():
    # Create the output directories if they don't exist
    os.makedirs("data/left", exist_ok=True)
    os.makedirs("data/right", exist_ok=True)
    
    # Get all available cameras
    available_cameras = pylon.TlFactory.GetInstance().EnumerateDevices()
    if len(available_cameras) < 2:
        print(f"Error: Found only {len(available_cameras)} camera(s). Need at least 2 cameras.")
        return
    
    # Connect to the first two available cameras
    right_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(available_cameras[1]))
    left_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(available_cameras[0]))
    
    # Open cameras
    left_camera.Open()
    right_camera.Open()

    exposure_time = 35000
    left_camera.ExposureTime.SetValue(exposure_time)
    right_camera.ExposureTime.SetValue(exposure_time)

    camera_width = 3072
    left_camera.Width.SetValue(camera_width)
    right_camera.Width.SetValue(camera_width)

    left_offset_x = 864
    left_camera.OffsetX.SetValue(left_offset_x)
    right_offset_x = 564
    right_camera.OffsetX.SetValue(right_offset_x)
    
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
        left_result = left_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        right_result = right_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
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
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                left_filename = f"data/left/left_{timestamp}_{frame_count}.png"
                right_filename = f"data/right/right_{timestamp}_{frame_count}.png"
                
                cv2.imwrite(left_filename, left_img)
                cv2.imwrite(right_filename, right_img)
                
                print(f"Captured frame {frame_count} to {left_filename} and {right_filename}")
                frame_count += 1
            
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
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()