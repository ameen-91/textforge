import os
import sys
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from textforge.utils import (
    print_success_bold,
    get_memory_usage,
    print_neutral,
    unsanitize_model_name,
    get_package_dir,
)
import onnxruntime
from textforge.base import PipelineStep


class DeploymentStep(PipelineStep):
    """Pipeline step for deploying a model using FastAPI and ONNX Runtime."""

    def __init__(self):
        """Initialize DeploymentStep."""
        super().__init__()

    def run(self, model_path, quantize: bool = False):
        """Run the deployment process.

        Args:
            model_path (str): Path to the model directory.
            quantize (bool, optional): Flag to indicate whether to quantize the model. Defaults to False.
        """
        quantize = "True" if quantize else "False"
        subprocess.run(
            [
                "python",
                os.path.join(get_package_dir(), "serve.py"),
                os.path.join(model_path, "model"),
                str(quantize).lower(),
            ]
        )

    def save(self):
        """Save deployment configuration or artifacts.

        Note:
            This method is currently a placeholder.
        """
        pass
