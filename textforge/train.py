from copy import deepcopy
import os
from textforge.base import PipelineStep
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
import evaluate
import re
from datasets import Dataset
import torch
from transformers.utils import logging
from rich.console import Console
from rich.table import Table


class CustomCallback(TrainerCallback):
    """Custom callback to evaluate the model on the training dataset at the end of each epoch."""

    def __init__(self, trainer) -> None:
        """
        Args:
            trainer (Trainer): The Trainer instance.
        """
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): The state of the trainer.
            control (TrainerControl): The control object.
            **kwargs: Additional keyword arguments.

        Returns:
            TrainerControl: The control object.
        """
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


class TrainingStep(PipelineStep):
    """Pipeline step for training a sequence classification model."""

    def __init__(
        self,
        model_name="distilbert/distilbert-base-uncased",
        batch_size=8,
        epochs=3,
        lr=5e-5,
        max_length=128,
        save_steps=500,
        eval_steps=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=None,
    ):
        """
        Args:
            model_name (str): The name of the pre-trained model.
            batch_size (int): The batch size for training and evaluation.
            epochs (int): The number of training epochs.
            lr (float): The learning rate.
            max_length (int): The maximum sequence length.
            save_steps (int): The number of steps between model saves.
            eval_steps (int): The number of steps between evaluations.
            device (str): The device to use for training (e.g., 'cuda' or 'cpu').
            model_path (str, optional): The path to a pre-trained model.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.device = device
        self.max_length = max_length
        self.model_path = model_path
        self.console = Console()

    def print_config_options(self):
        """Prints the configuration options for the training step."""
        table = Table(title="TrainingStep Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")

        options = {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.learning_rate,
            "max_length": self.max_length,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "device": self.device,
        }

        for key, value in options.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def run(self, data):
        """
        Runs the training step.

        Args:
            data (pandas.DataFrame): The input data containing 'text' and 'label' columns.

        Returns:
            AutoModelForSequenceClassification: The trained model.
        """
        labels = data["label"].unique()
        num_labels = len(labels)
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}

        def clean_paragraph(paragraph):
            # Remove all symbols except for alphanumeric characters and spaces
            paragraph = re.sub(r"[^a-zA-Z0-9\s]", "", paragraph)
            # Replace multiple spaces with a single space
            paragraph = re.sub(r"\s+", " ", paragraph)
            # Strip leading and trailing spaces
            paragraph = paragraph.strip()

            return paragraph

        def clean(examples):
            examples["text"] = clean_paragraph(examples["text"])
            return examples

        def load_data(data):
            # Load the data from the input path
            # data = pd.read_csv(input_path)
            data = data.astype({"label": int})
            data = Dataset.from_pandas(data)
            data = data.map(clean)
            data = data.class_encode_column("label")
            return data

        data["label"] = data["label"].map(lambda x: label2id[x])
        data = load_data(data)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        def tokenize(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        tokenized_data = data.map(tokenize, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tokenized_data = tokenized_data.train_test_split(test_size=0.3)
        if self.model_path:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS,
        #     inference_mode=False,
        #     r=4,
        #     lora_alpha=16,
        #     lora_dropout=0.05,
        #     target_modules="all-linear",
        #     modules_to_save=["classifier"]
        # )
        # model = get_peft_model(model=model, peft_config=peft_config)
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir="output",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=2,
            report_to="none",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            data_collator=data_collator,
        )
        trainer.add_callback(CustomCallback(trainer))
        trainer.train()
        return model

    def save(self, model, output_path):
        """
        Saves the trained model and tokenizer.

        Args:
            model (AutoModelForSequenceClassification): The trained model.
            output_path (str): The path to save the model and tokenizer.
        """
        model.save_pretrained(os.path.join(output_path, "model"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.save_pretrained(os.path.join(output_path, "model"))
