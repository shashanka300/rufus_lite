"""
Thin wrapper around the local Ollama API.
Supports both streaming and non-streaming chat.
"""

from __future__ import annotations

from collections.abc import Iterator

import ollama

DEFAULT_MODEL = "qwen3.5:latest"


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def chat(self, messages: list[dict], stream: bool = False) -> str | Iterator[str]:
        if stream:
            return self._stream(messages)
        response = ollama.chat(model=self.model, messages=messages)
        return response.message.content

    def _stream(self, messages: list[dict]) -> Iterator[str]:
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            if chunk.message.content:
                yield chunk.message.content
