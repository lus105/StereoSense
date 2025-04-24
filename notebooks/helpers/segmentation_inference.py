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

    def __call__(self, data: np.ndarray) -> np.ndarray:
        try:
            input_name = self.model.get_inputs()[0].name
            output_names = [output.name for output in self.model.get_outputs()]
            outputs = self.model.run(output_names, {input_name: data})

            return outputs
        except Exception as e:
            raise RuntimeError(f'ONNX prediction failed: {e}')
