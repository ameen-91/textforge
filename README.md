# Infero

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/textforge)
![PyPI - Version](https://img.shields.io/pypi/v/textforge)
![PyPI - Downloads](https://img.shields.io/pypi/dw/textforge)
[![CI](https://github.com/ameen-91/textforge/actions/workflows/ci.yaml/badge.svg)](https://github.com/norsulabs/textforge/actions/workflows/ci.yaml)
![Coverage](static/coverage.svg)

## Overview

TextForge automates model distillation, training, quantization, and deployment for text classification. It simplifies synthetic data generation, model optimization using ONNX runtime, and FastAPI serving.

### Features

- Automated synthetic data generation
- Transformer model training
- ONNX conversion with 8-bit quantization
- Automated model API serving with FastAPI
<!-- - Customizable hyperparameter control -->

### Installation

```bash
pip install textforge
```

## Usage

```python
import pandas as pd
from textforge.pipeline import Pipeline, PipelineConfig

pipeline_config = PipelineConfig(
    api_key=api_key,
    labels=['business','education','entertainment','sports','technology'],
    query="Classify based on headlines",
    save_steps=200,
    eval_steps=200,
    epochs=10
)

df = pd.read_csv('data.csv')

pipeline = Pipeline(pipeline_config)

pipeline.run(data=df, save=True, serve=True)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
