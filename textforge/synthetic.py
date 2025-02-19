import os
import asyncio
import time
import pandas as pd
from tqdm import tqdm  # note: switched from tqdm.asyncio to synchronous tqdm
from IPython import get_ipython

from textforge.base import PipelineStep
from textforge.utils import extract_label_value
from openai import AsyncClient, Client as SyncClient  # Using SyncClient for sync calls


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
        async with self._rate_lock:
            current_time = asyncio.get_event_loop().time()
            delay = self.rate_limit_interval - (current_time - self._last_request_time)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_time = asyncio.get_event_loop().time()

    def _throttle_sync(self):
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
                    **options,
                )
                response = extract_label_value(response_obj.choices[0].message.content)
                tries -= 1
            return response

        texts = labelled_data[labelled_data.columns[0]].tolist()
        tasks = [asyncio.create_task(classify_text(text)) for text in texts]
        results = []
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Classifying text"
        ):
            results.append(await task)
        labelled_data["label"] = results
        labelled_data.rename(columns={labelled_data.columns[0]: "text"}, inplace=True)
        self.print_stats(labelled_data)
        return labelled_data

    def run(
        self, data: pd.DataFrame, max_tokens: int = None, max_tries: int = 5
    ) -> pd.DataFrame:
        """
        Executes the pipeline. If the instance was initialized with sync_client=True,
        the synchronous pipeline (run_sync) is executed. Otherwise the asynchronous pipeline (run_async) is used.
        """
        if self.sync_client:
            return self.run_sync(data, max_tokens=max_tokens, max_tries=max_tries)
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.run_async(data, max_tokens, max_tries)
                )
            else:
                return asyncio.run(self.run_async(data, max_tokens, max_tries))
        except NameError:
            return asyncio.run(self.run_async(data, max_tokens, max_tries))

    def run_sync(
        self,
        data: pd.DataFrame,
        max_tokens: int = None,
        max_tries: int = 10,
    ) -> pd.DataFrame:
        """
        Synchronous pipeline for text classification.
        Processes the data synchronously without using asyncio.
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
        for text in tqdm(texts, desc="Classifying text"):
            result = classify_text_sync(text)
            results.append(result)
        labelled_data["label"] = results
        labelled_data.rename(columns={labelled_data.columns[0]: "text"}, inplace=True)
        self.print_stats(labelled_data)
        return labelled_data

    def save(self, data: pd.DataFrame, output_path: str):
        data.to_csv(os.path.join(output_path, "labelled_data.csv"), index=False)

    def print_stats(self, data: pd.DataFrame):
        print(f"Total number of samples: {len(data)}")
        print(f"Number of unique labels: {data['label'].nunique()}")
        print(f"Labels: {data['label'].unique()}")
        if "label" in data.columns:
            print(
                f"Label distribution: {data['label'].value_counts() / len(data) * 100}"
            )
