import time

import torch
import numpy as np
import onnxruntime


class StereoInferenceOnnx:
    """Basic inference implementation for stereo ONNX models using ONNX Runtime."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self.model = None
        self.initialize()

    def initialize(self) -> None:
        try:
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Set execution provider options to reduce memory copies
            provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }

            self.model = onnxruntime.InferenceSession(
                self._model_path,
                sess_options=session_options,
                providers=[('CUDAExecutionProvider', provider_options)],
            )

            # Get input and output information
            self.input_names = [input.name for input in self.model.get_inputs()]
            self.output_names = [output.name for output in self.model.get_outputs()]

            print('Model initialized successfully')
            print(f'Input names: {self.input_names}')
            print(f'Output names: {self.output_names}')

        except Exception as e:
            raise RuntimeError(f'Failed to load ONNX model: {e}')

    def __call__(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Run inference on stereo image pair.

        Args:
            left_img: Left image in NCHW format [1, 3, H, W] with float32 values
            right_img: Right image in NCHW format [1, 3, H, W] with float32 values

        Returns:
            Disparity map as numpy array
        """
        try:
            # Create input dictionary with both images
            input_dict = {self.input_names[0]: left_img, self.input_names[1]: right_img}

            # Run inference
            outputs = self.model.run(self.output_names, input_dict)

            # Return the first output (disparity map)
            result = outputs[0].copy()

            return result

        except Exception as e:
            raise RuntimeError(f'ONNX prediction failed: {e}')


def main():
    inference_engine = StereoInferenceOnnx('models/fs_800_640.onnx')

    left_img = np.zeros((1, 3, 800, 640), dtype=np.float32)
    right_img = np.zeros((1, 3, 800, 640), dtype=np.float32)

    # Run inference
    start_time = time.time()
    disparity = inference_engine(left_img, right_img)
    print(f'Disparity shape: {disparity.shape}')
    end_time = time.time()
    print(f'Infe_rence time: {end_time - start_time:.4f} seconds')


if __name__ == '__main__':
    main()
