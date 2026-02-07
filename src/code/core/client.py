"""Unified LLM client supporting multiple providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass(slots=True)
class LlmConfig:
    """Configuration for LLM client.

    Attributes:
        model: Model name
        temperature: Sampling temperature (0.0 to 1.0)
        api_key: API key (defaults to API_KEY env var)
        base_url: API base URL (defaults to BASE_URL env var)
        max_retries: Maximum retry attempts on failure
        retry_min_wait: Minimum wait time between retries (seconds)
        retry_max_wait: Maximum wait time between retries (seconds)
    """

    model: str = "gpt-5-mini"
    temperature: float = 0.3
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 3
    retry_min_wait: int = 1
    retry_max_wait: int = 5
    max_workers: int = 5

    def __post_init__(self) -> None:
        # Auto-load from environment variables if not provided
        load_dotenv()
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY", "")
        if self.base_url is None:
            self.base_url = os.getenv("BASE_URL", "")


class LlmClient:
    def __init__(self, config: LlmConfig | None = None) -> None:
        """Initialize LLM client with configuration.

        Args:
            config: LLM configuration (uses defaults if None)
        """
        self.config = config or LlmConfig()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def complete(self, prompt: str, *, options: Mapping[str, Any] | None = None) -> str:
        """Generate completion for the given prompt.

        Args:
            prompt: Input prompt text
            options: Optional override parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            ValueError: If API returns empty response
            Exception: If all retry attempts fail
        """
        temperature = (
            options.get("temperature", self.config.temperature)
            if options
            else self.config.temperature
        )

        # Use tenacity retry decorator
        @retry(
            wait=wait_random_exponential(
                min=self.config.retry_min_wait, max=self.config.retry_max_wait
            ),
            stop=stop_after_attempt(self.config.max_retries),
        )
        def _generate() -> str:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            content = response.choices[0].message.content
            if content:
                return content
            else:
                raise ValueError("Empty response from API")

        return _generate()

    def batch_complete(
        self,
        prompts: Iterable[str],
        *,
        options: Mapping[str, Any] | None = None,
        max_workers: int | None = None,
    ) -> list[str | None]:
        """Generate completions for multiple prompts in parallel.

        Returns a list aligned to the input order. Failed items return None.
        """
        prompt_list = list(prompts)
        if not prompt_list:
            return []

        worker_count = max_workers or self.config.max_workers
        results: list[str | None] = [None] * len(prompt_list)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(self.complete, prompt, options=options): idx
                for idx, prompt in enumerate(prompt_list)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None

        return results
