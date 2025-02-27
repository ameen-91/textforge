import os
import sys
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from utils import (
    print_success_bold,
    get_memory_usage,
    print_neutral,
    unsanitize_model_name,
    get_package_dir,
)
import onnxruntime
from textforge.base import PipelineStep


class DeploymentStep(PipelineStep):

    def __init__(self):
        super().__init__()

    def run(self, model_path, quantize: bool = False):
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
        pass
