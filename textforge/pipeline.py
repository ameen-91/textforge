import os
from textforge.base import PipelineStep
from textforge.synthetic import SyntheticDataGeneration
from textforge.train import TrainingStep


class PipelineConfig:
    def __init__(
        self,
        api_key,
        labels,
        query,
        data_gen_model="gpt-4o-mini",
        model="distilbert/distilbert-base-uncased",
        max_length=128,
        epochs=3,
        batch_size=8,
    ):
        self.api_key = api_key
        self.labels = labels
        self.query = query
        self.data_gen_model = data_gen_model
        self.model = model
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.steps = [
            SyntheticDataGeneration(
                api_key=config.api_key,
                labels=config.labels,
                query=config.query,
                model=config.data_gen_model,
            ),
            TrainingStep(
                model=config.model,
                max_length=config.max_length,
                epochs=config.epochs,
                batch_size=config.batch_size,
            ),
        ]
        for step in self.steps:
            if hasattr(step, "print_config_options"):
                step.print_config_options()

    def run(self, data, save=False):
        data = data.copy()
        for step in self.steps:
            data = step.run(data)
            if save:
                step.save(data)
        return data
