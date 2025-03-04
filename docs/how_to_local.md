# Using TextForge with a Local LLM

## Installation

First, install Ollama to enable local LLM support:

```python
from textforge.textforge.utils import install_ollama
install_ollama(model="llama3.1:8b-instruct-q4_0") # Replace with desired model or leave blank for default
```

## Basic Usage

Here's a complete example of using TextForge with a local LLM:

```python
from textforge.pipeline import PipelineConfig, Pipeline
import pandas as pd

# Configure the pipeline
pipeline_config = PipelineConfig(
    use_local=True,
    data_gen_model="llama3.1:8b-instruct-q4_0",
    labels=['business', 'education', 'entertainment', 'sports', 'technology'],
    query="Classify based on headlines",
    epochs=30,
    model_name="distilbert-base-uncased",
    save_steps=200,
    eval_steps=200
)

# Create and run pipeline
pipe = Pipeline(config=pipeline_config)
data = pd.read_csv("data.csv")
models = pipe.run(data=data, save=True)
```