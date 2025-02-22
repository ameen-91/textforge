import os
import time
from textforge.base import PipelineStep
from textforge.synthetic import SyntheticDataGeneration
from textforge.train import TrainingStep
from textforge.quantize import QuantizeStep


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
        sync_client=True,
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
        self.sync_client = sync_client


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.step1 = SyntheticDataGeneration(
            api_key=config.api_key,
            labels=config.labels,
            query=config.query,
            model=config.data_gen_model,
            base_url=config.base_url,
            sync_client=config.sync_client,
        )
        self.step2 = TrainingStep(
            model_name=config.model_name,
            max_length=config.max_length,
            epochs=config.epochs,
            batch_size=config.batch_size,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            model_path=config.model_path,
        )
        self.step3 = QuantizeStep()

        if hasattr(self.step1, "print_config_options"):
            self.step1.print_config_options()
        if hasattr(self.step2, "print_config_options"):
            self.step2.print_config_options()
        if hasattr(self.step3, "print_config_options"):
            self.step3.print_config_options()

    def run(self, data, save=False, skip_data_generation=False):
        run_id = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"outputs/{run_id}/"
        os.makedirs(output_path, exist_ok=True)

        data = data.copy()

        if not skip_data_generation:
            data = self.step1.run(data)
            if save:
                self.step1.save(data, output_path)

        data = self.step2.run(data)
        if save:
            self.step2.save(data, output_path)

        data = self.step3.run(output_path)

        return output_path
