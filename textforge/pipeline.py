import os
import time
from textforge.base import PipelineStep
from textforge.synthetic import SyntheticDataGeneration
from textforge.train import TrainingStep
from textforge.quantize import QuantizeStep
from textforge.deployment import DeploymentStep


class PipelineConfig:
    """Configuration class for the pipeline."""

    def __init__(
        self,
        labels,
        query,
        api_key=None,
        use_local=False,
        data_gen_model="gpt-4o-mini",
        model_name="distilbert/distilbert-base-uncased",
        model_path=None,
        max_length=128,
        epochs=3,
        batch_size=8,
        save_steps=100,
        eval_steps=100,
        base_url=None,
        sync_client=False,
    ):
        """
        Args:
            api_key (str): API key for data generation.
            labels (list): List of labels for classification.
            query (str): Query for data generation.
            use_local (bool): Whether to use local data generation.
            data_gen_model (str): Model name for synthetic data generation.
            model_name (str): Model name for training.
            model_path (str, optional): Path to a pre-trained model.
            max_length (int): Maximum sequence length.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and evaluation.
            save_steps (int): Number of steps between model saves.
            eval_steps (int): Number of steps between evaluations.
            base_url (str, optional): Base URL for API requests.
            sync_client (bool): Whether to use a synchronous client.
        """
        if use_local is False and api_key is None:
            raise ValueError("API key is required for remote data generation.")
        if use_local and data_gen_model == "gpt-4o-mini":
            raise ValueError(
                "Local data generation is not supported for GPT-4o-mini. Please use a different model."
            )

        if use_local:
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"

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
    """Pipeline for synthetic data generation, training, quantization, and deployment."""

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config (PipelineConfig): Configuration for the pipeline.
        """
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

        self.step4 = DeploymentStep()

        if hasattr(self.step1, "print_config_options"):
            self.step1.print_config_options()
        if hasattr(self.step2, "print_config_options"):
            self.step2.print_config_options()
        if hasattr(self.step3, "print_config_options"):
            self.step3.print_config_options()

    def run(self, data, serve=False, save=False, skip_data_generation=False):
        """
        Runs the pipeline.

        Args:
            data (pandas.DataFrame): The input data.
            serve (bool): Whether to serve the model after training.
            save (bool): Whether to save the intermediate and final outputs.
            skip_data_generation (bool): Whether to skip the data generation step.

        Returns:
            str: The output path where the results are saved.
        """
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
        if serve:
            self.step4.run(data)

        return output_path
