#!/usr/bin/env python3
"""LLaMA-Factory WebUI (Gradio) client for LoRA inference."""

from __future__ import annotations

import os
import time

from gradio_client import Client

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_BASE_MODEL = os.environ.get("CNC_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_CHECKPOINT = "cnc-qwen2.5-7b-lora"


def last_assistant_text(history) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            return str(msg[1])
    last = history[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


class LlmFactoryClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        try:
            self.client = Client(base_url)
        except ValueError as exc:
            raise ConnectionError(
                f"无法连接 LLaMA-Factory WebUI ({base_url})。"
                "请先启动 WebUI 并在 Chat 页加载 LoRA，或使用 --skip-load。"
            ) from exc

    def load_lora(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        base_model: str = DEFAULT_BASE_MODEL,
        wait_seconds: int = 90,
    ) -> None:
        result = self.client.predict(
            "en",
            "Qwen2.5-7B-Instruct",
            base_model,
            "lora",
            [checkpoint],
            "none",
            "bnb",
            "default",
            "none",
            "auto",
            "huggingface",
            "auto",
            '{"vllm_enforce_eager": true}',
            api_name="/load_model",
        )
        if isinstance(result, str) and "unload" in result.lower():
            return
        time.sleep(wait_seconds)

    def chat(
        self,
        query: str,
        system: str = "",
        max_new_tokens: int = 768,
        temperature: float = 0.3,
    ) -> str:
        append_result = self.client.predict([], "user", query, True, api_name="/append")
        chatbot = append_result[0] if isinstance(append_result, tuple) else append_result
        final = None
        for update in self.client.submit(
            chatbot,
            "en",
            system,
            "",
            None,
            None,
            None,
            max_new_tokens,
            0.7,
            temperature,
            True,
            True,
            False,
            api_name="/stream",
        ):
            final = update
        return last_assistant_text(final)
