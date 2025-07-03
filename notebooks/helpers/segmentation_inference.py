import os
import rootutils
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
