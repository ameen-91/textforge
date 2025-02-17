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
        model_name="distilbert/distilbert-base-uncased",
        model_path=None,
        max_length=128,
        epochs=3,
        batch_size=8,
        save_steps=100,
        eval_steps=100,
        base_url=None,
    ):
        self.api_key = api_key
        self.labels = labels
        self.query = query
        self.data_gen_model = data_gen_model
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.model_path = model_path
        self.base_url = base_url


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.steps = [
            SyntheticDataGeneration(
                api_key=config.api_key,
                labels=config.labels,
                query=config.query,
                model=config.data_gen_model,
                base_url=config.base_url,
            ),
            TrainingStep(
                model_name=config.model_name,
                max_length=config.max_length,
                epochs=config.epochs,
                batch_size=config.batch_size,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                model_path=config.model_path,
            ),
        ]
        for step in self.steps:
            if hasattr(step, "print_config_options"):
                step.print_config_options()

    def run(self, data, save=False, only_train=False):
        data = data.copy()
        if only_train:
            self.steps = self.steps[1:]
        for step in self.steps:
            data = step.run(data)
            if save:
                step.save(data)
        return data
