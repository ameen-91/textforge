import os
from textforge.base import PipelineStep
from textforge.utils import (
    sanitize_model_name,
    print_success,
    print_neutral,
    get_models_dir,
    print_success_bold,
    get_memory_usage,
    print_neutral,
    unsanitize_model_name,
)
from transformers import AutoModelForSequenceClassification
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import warnings
import logging

logging.getLogger("root").setLevel(
    logging.ERROR
)  ## temporary fix for warning message from onnxruntime


class QuantizeStep(PipelineStep):
    """Pipeline step for quantizing a model using ONNX Runtime."""

    def __init__(self):
        """Initialize QuantizeStep."""
        pass

    def convert_to_onnx(self, output_path):
        """Convert the model to ONNX format.

        Args:
            output_path (str): Directory containing the model to be converted.
        """
        onnx_path = os.path.join(output_path, "model", "model.onnx")

        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(output_path, "model")
        )

        with torch.inference_mode():
            inputs = {
                "input_ids": torch.ones(1, 512, dtype=torch.int64),
                "attention_mask": torch.ones(1, 512, dtype=torch.int64),
            }
            symbolic_names = {0: "batch_size", 1: "max_seq_len"}
            torch.onnx.export(
                model,
                (
                    inputs["input_ids"],
                    inputs["attention_mask"],
                ),
                onnx_path,
                opset_version=14,
                input_names=[
                    "input_ids",
                    "attention_mask",
                ],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                },
            )

    def convert_to_onnx_q8(self, output_path):
        """Quantize the ONNX model to 8-bit precision.

        Args:
            output_path (str): Directory containing the ONNX model.
        """
        onnx_model_path = os.path.join(output_path, "model/model.onnx")
        quantized_model_path = os.path.join(output_path, "model/model_quantized.onnx")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Please consider to run pre-processing before quantization",
            )
            quantize_dynamic(
                onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8
            )

    def run(self, output_path):
        """Run the quantization pipeline.

        Args:
            output_path (str): Directory where the model files are located.

        Returns:
            str: The output path after quantization is complete.
        """
        self.convert_to_onnx(output_path=output_path)
        self.convert_to_onnx_q8(output_path=output_path)
        print_success_bold("Quantization complete")
        return output_path

    def save(self, model):
        """Save the quantized model.

        Args:
            model: The quantized model to be saved.
        """
        pass
