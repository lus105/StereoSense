import os
import rootutils

import cv2
import numpy as np
import onnxruntime as ort

rootutils.setup_root(
    os.path.abspath(''), indicator=['.git', 'pyproject.toml'], pythonpath=True
)

from src.base_inference import InferenceBase

class SegmentationInferenceOnnx(InferenceBase):
    def initialize(self) -> None:
        try:
            self.model = ort.InferenceSession(
                self._model_path, providers=['CUDAExecutionProvider']
            )
        except Exception as e:
            raise RuntimeError(f'Failed to load ONNX model: {e}')
        
    @property
    def input_height(self) -> int:
        """Get the height dimension from the model input shape.

        Returns:
            The height of the input as an integer.
        """
        if self.model is None:
            raise RuntimeError('Model not initialized')
        # ONNX input shape is usually [batch, channels, height, width]
        input_shape = self.model.get_inputs()[0].shape
        # Check if the shape is dynamic or static
        if isinstance(input_shape[2], int):
            return input_shape[2]  # Return the height dimension
        else:
            # If dynamic, return None or raise an exception
            raise RuntimeError('Model has dynamic input height')

    @property
    def input_width(self) -> int:
        """Get the width dimension from the model input shape.

        Returns:
            The width of the input as an integer.
        """
        if self.model is None:
            raise RuntimeError('Model not initialized')
        # ONNX input shape is usually [batch, channels, height, width]
        input_shape = self.model.get_inputs()[0].shape
        # Check if the shape is dynamic or static
        if isinstance(input_shape[3], int):
            return input_shape[3]  # Return the width dimension
        else:
            # If dynamic, return None or raise an exception
            raise RuntimeError('Model has dynamic input width')

    def __call__(self, data: np.ndarray) -> np.ndarray:
        try:
            input_name = self.model.get_inputs()[0].name
            output_names = [output.name for output in self.model.get_outputs()]
            outputs = self.model.run(output_names, {input_name: data})

            return outputs
        except Exception as e:
            raise RuntimeError(f'ONNX prediction failed: {e}')


def segmentation_preprocess(image:np.ndarray, model: SegmentationInferenceOnnx) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model.input_width, model.input_height))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image_tensor = np.expand_dims(image, axis=0)

    return image_tensor


def segmentation_postprocess(mask: np.ndarray) -> np.ndarray:
    # Morphological erosion to clean up the mask
    kernel_size = 5
    iterations = 6
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=iterations)
    
    # Find connected components and keep only the largest one
    mask_uint8 = mask_eroded.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    # Find the largest component (excluding background which is label 0)
    if num_labels > 1:
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_largest_component = (labels == largest_component_label).astype(np.uint8) * 255
    else:
        mask_largest_component = np.zeros_like(mask_uint8)

    # Flood fill from border pixels to fill holes
    h, w = mask_largest_component.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    mask_copy = mask_largest_component.copy()
    
    # Flood fill from all border pixels starting at (0, 0)
    cv2.floodFill(mask_copy, flood_fill_mask, (0, 0), 255)
    
    # Invert the flood filled image and combine with original
    mask_inv = cv2.bitwise_not(mask_copy)
    mask_filled = mask_largest_component | mask_inv
    
    # Convert back to float32 for consistency
    mask_final = mask_filled.astype(np.float32)

    return mask_final


def segmentation_inference(
    model: SegmentationInferenceOnnx,
    image: np.ndarray,
) -> np.ndarray:
    """Perform segmentation inference on an input image and optionally rectify the output mask.

    Args:
        model (SegmentationInferenceOnnx): The segmentation model for inference.
        image (np.ndarray): Input image for segmentation.

    Returns:
        np.ndarray: Output segmentation mask.
    """
    # Preprocess the image
    image_proc = image.copy()
    image_tensor = segmentation_preprocess(image_proc, model)
    # Perform inference
    mask = model(image_tensor)

    # Postprocess the output
    mask = np.squeeze(mask)
    mask = 1 / (1 + np.exp(-mask))
    mask = (mask > 0.5).astype(np.float32) * 255
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    
    mask = segmentation_postprocess(mask)

    return mask