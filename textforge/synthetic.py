import os
import asyncio
import time
import pandas as pd
from tqdm import tqdm  # note: switched from tqdm.asyncio to synchronous tqdm
from IPython import get_ipython

from textforge.base import PipelineStep
from textforge.utils import extract_label_value
from openai import AsyncClient, Client as SyncClient  # Using SyncClient for sync calls
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn


class SyntheticDataGeneration(PipelineStep):
    def __init__(
        self,
        api_key: str,
        labels: list[str],
        query: str = "",
        model: str = "gpt-4o-mini",
        rate_limit_interval: float = 0.2,
        base_url=None,
        sync_client: bool = False,  # new flag to choose client type
    ):
        """Initialize SyntheticDataGeneration.

        Args:
            api_key (str): API key for authentication.
            labels (list[str]): List of labels for classification.
            query (str, optional): Additional query context. Defaults to "".
            model (str, optional): Model name to use. Defaults to "gpt-4o-mini".
            rate_limit_interval (float, optional): Interval between API calls. Defaults to 0.2.
            base_url (optional): Base URL for API endpoint.
            sync_client (bool, optional): Flag to choose synchronous client. Defaults to False.
        """
        self.base_url = base_url
        self.sync_client = sync_client
        if self.sync_client:
            if base_url:
                self.client = SyncClient(api_key=api_key, base_url=base_url)
            else:
                self.client = SyncClient(api_key=api_key)
        else:
            if base_url:
                self.client = AsyncClient(api_key=api_key, base_url=base_url)
            else:
                self.client = AsyncClient(api_key=api_key)
        self.model = model
        self.labels = labels
        self.query = query
        self.rate_limit_interval = rate_limit_interval
        # Async rate throttling helpers
        self._rate_lock = asyncio.Lock()
        self._last_request_time = 0
        # For synchronous throttling we'll use time.time()
        self._last_sync_request_time = time.time()

    async def _throttle(self):
        """Asynchronously throttle API calls based on rate_limit_interval."""
        async with self._rate_lock:
            current_time = asyncio.get_event_loop().time()
            delay = self.rate_limit_interval - (current_time - self._last_request_time)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_time = asyncio.get_event_loop().time()

    def _throttle_sync(self):
        """Synchronously throttle API calls based on rate_limit_interval."""
        current_time = time.time()
        delay = self.rate_limit_interval - (current_time - self._last_sync_request_time)
        if delay > 0:
            time.sleep(delay)
        self._last_sync_request_time = time.time()

    async def generate_text(
        self,
        data: pd.DataFrame,
        system_prompt: str = "You are a helpful AI assistant. Please provide a response to the following user query:",
        max_tokens: int = None,
    ) -> pd.DataFrame:
        """Generate text for each row in the provided DataFrame asynchronously.

        Args:
            data (pd.DataFrame): Input data with text input in the first column.
            system_prompt (str, optional): System prompt for the API. Defaults to a helpful assistant prompt.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with an added 'output' column containing generated responses.
        """
        labelled_data = data.copy()

        async def generate_response(text):
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            await self._throttle()
            response_obj = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "assistant", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                **options,
            )
            return response_obj.choices[0].message.content

        texts = labelled_data[labelled_data.columns[0]].tolist()
        tasks = [asyncio.create_task(generate_response(text)) for text in texts]
        responses = []
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating text"
        ):
            responses.append(await task)
        labelled_data["output"] = responses
        return labelled_data

    def create_system_prompt(self, labels: list[str], query: str = "") -> str:
        """Create a system prompt for text classification.

        Args:
            labels (list[str]): List of classification labels.
            query (str, optional): Additional query to refine prompt. Defaults to "".

        Returns:
            str: Constructed system prompt.
        """
        labels_str = ", ".join(labels)
        if query:
            return (
                f"Classify the following text into one of the following categories: {labels_str} "
                f"based on {query}. Answer in JSON Format. Format: {{'label':'ans'}}. Absolutely no context is needed."
            )
        else:
            return (
                f"Classify the following text into one of the following categories: {labels_str}. "
                "Answer in JSON Format. Format: {'label':'ans'}. Absolutely no context is needed."
            )

    async def run_async(
        self,
        data: pd.DataFrame,
        max_tokens: int = None,
        max_tries: int = 5,
    ) -> pd.DataFrame:
        """Run the asynchronous text classification pipeline.

        Args:
            data (pd.DataFrame): Input data with text input.
            max_tokens (int, optional): Maximum tokens for generation. Defaults to None.
            max_tries (int, optional): Maximum reattempts for valid classification. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame with classification results added.
        """
        labelled_data = data.copy()
        system_prompt = self.create_system_prompt(self.labels, self.query)

        async def classify_text(text):
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            await self._throttle()
            response_obj = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                **options,
            )
            response = extract_label_value(response_obj.choices[0].message.content)
            tries = max_tries
            while response not in self.labels and tries > 0:
                await self._throttle()
                response_obj = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You did not respond in JSON Format. Format: {'label':'ans'}. Absolutely no context is needed."
                            + system_prompt,
                        },
                        {"role": "user", "content": text},
                    ],
                )
                response = extract_label_value(response_obj.choices[0].message.content)
                tries -= 1
            return response

        texts = labelled_data[labelled_data.columns[0]].tolist()

        results = [None] * len(texts)

        async def progress_wrapper(idx, text, task_id, progress):
            result = await classify_text(text)
            results[idx] = result
            progress.update(task_id, advance=1)

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        with progress:
            task_id = progress.add_task("Classifying text", total=len(texts))
            await asyncio.gather(
                *(
                    progress_wrapper(i, text, task_id, progress)
                    for i, text in enumerate(texts)
                )
            )
        labelled_data["label"] = results
        labelled_data.rename(columns={labelled_data.columns[0]: "text"}, inplace=True)
        self.print_stats(labelled_data)
        return labelled_data

    def run(
        self, data: pd.DataFrame, max_tokens: int = None, max_tries: int = 5
    ) -> pd.DataFrame:
        """Execute the pipeline synchronously or asynchronously based on client type.

        Args:
            data (pd.DataFrame): Input data with text.
            max_tokens (int, optional): Maximum tokens per API call. Defaults to None.
            max_tries (int, optional): Maximum retries for valid classification. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame with processed results.
        """
        if self.sync_client:
            return self.run_sync(data, max_tokens=max_tokens, max_tries=max_tries)
        try:
            shell = get_ipython().__class__.__name__
            if shell in ["ZMQInteractiveShell", "Shell", "Google.Colab"]:
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.run_async(data, max_tokens, max_tries)
                )
        except NameError:
            pass
        else:
            return asyncio.run(self.run_async(data, max_tokens, max_tries))

    def run_sync(
        self,
        data: pd.DataFrame,
        max_tokens: int = None,
        max_tries: int = 10,
    ) -> pd.DataFrame:
        """Run the synchronous text classification pipeline.

        Args:
            data (pd.DataFrame): Input data with text.
            max_tokens (int, optional): Maximum tokens for generation. Defaults to None.
            max_tries (int, optional): Maximum reattempts for valid classification. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame with classification results added.
        """
        labelled_data = data.copy()
        system_prompt = self.create_system_prompt(self.labels, self.query)

        def classify_text_sync(text):
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            self._throttle_sync()
            response_obj = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                **options,
            )
            response = extract_label_value(response_obj.choices[0].message.content)
            tries = max_tries
            while response not in self.labels and tries > 0:
                self._throttle_sync()
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You did not respond in JSON Format. Format: {'label':'ans'}. Absolutely no context is needed."
                            + system_prompt,
                        },
                        {"role": "user", "content": text},
                    ],
                    **options,
                )
                response = extract_label_value(response_obj.choices[0].message.content)
                tries -= 1
            return response

        texts = labelled_data[labelled_data.columns[0]].tolist()
        results = []

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        with progress:
            task_id = progress.add_task("Classifying text", total=len(texts))
            for i, text in enumerate(texts):
                result = classify_text_sync(text)
                results.append(result)
                progress.update(task_id, advance=1)
        labelled_data["label"] = results
        labelled_data.rename(columns={labelled_data.columns[0]: "text"}, inplace=True)
        self.print_stats(labelled_data)
        return labelled_data

    def save(self, data: pd.DataFrame, output_path: str):
        """Save the labelled data to a CSV file.

        Args:
            data (pd.DataFrame): DataFrame containing processed data.
            output_path (str): Directory path where the file will be saved.
        """
        data.to_csv(os.path.join(output_path, "labelled_data.csv"), index=False)

    def print_stats(self, data: pd.DataFrame):
        """Print statistics about the labelled data.

        Args:
            data (pd.DataFrame): DataFrame containing the results.
        """
        print(f"Total number of samples: {len(data)}")
        print(f"Number of unique labels: {data['label'].nunique()}")
        print(f"Labels: {data['label'].unique()}")
        if "label" in data.columns:
            print(
                f"Label distribution: {data['label'].value_counts() / len(data) * 100}"
            )
