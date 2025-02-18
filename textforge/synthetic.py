import os
import asyncio
import time
import pandas as pd
from tqdm.asyncio import tqdm
from IPython import get_ipython

from textforge.base import PipelineStep
from openai import AsyncClient


class SyntheticDataGeneration(PipelineStep):
    def __init__(
        self,
        api_key: str,
        labels: list[str],
        query: str = "",
        model: str = "gpt-4o-mini",
        rate_limit_interval: float = 0.2,
        base_url=None,
    ):
        self.base_url = base_url
        if base_url:
            self.client = AsyncClient(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncClient(api_key=api_key)
        self.model = model
        self.labels = labels
        self.query = query
        self.rate_limit_interval = rate_limit_interval
        # asyncio rate throttling helpers
        self._rate_lock = asyncio.Lock()
        self._last_request_time = 0

    async def _throttle(self):
        async with self._rate_lock:
            current_time = asyncio.get_event_loop().time()
            delay = self.rate_limit_interval - (current_time - self._last_request_time)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_time = asyncio.get_event_loop().time()

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
                f"based on {query}. Just answer with the label. Absolutely no context is needed."
            )
        else:
            return (
                f"Classify the following text into one of the following categories: {labels_str}. "
                "Just answer with the label. Absolutely no context is needed."
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
            response = response_obj.choices[0].message.content
            tries = max_tries
            while response not in self.labels and tries > 0:
                await self._throttle()
                response_obj = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You did not respond with just the label please respond again with the label only. "
                                "Without any context or explanation. "
                            )
                            + system_prompt,
                        },
                        {"role": "user", "content": text},
                    ],
                    **options,
                )
                response = response_obj.choices[0].message.content
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
        return labelled_data

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.run_async(data))
            else:
                return asyncio.run(self.run_async(data))
        except NameError:
            return asyncio.run(self.run_async(data))

    def save(self, data: pd.DataFrame, output_path: str):
        data.to_csv(os.path.join(output_path, "labelled_data.csv"), index=False)
