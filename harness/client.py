"""Thin wrapper around DeepSeek's OpenAI-compatible chat completions API."""

import os
import re

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = (
    "You are a careful coding assistant. Solve the given task in the "
    "requested language. Respond with a single fenced code block "
    "containing only the final, runnable solution -- no explanations "
    "before or after the code block."
)

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and "
                "add your key from https://platform.deepseek.com/api_keys"
            )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-coder")
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def generate_code(self, prompt: str, language: str, temperature: float = 0.2) -> dict:
        """Ask the model to solve a task. Returns raw text + extracted code."""
        user_prompt = f"Language: {language}\n\nTask:\n{prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        code = self._extract_code(text)
        return {
            "raw_response": text,
            "code": code,
            "usage": dict(response.usage) if response.usage else None,
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        match = CODE_BLOCK_RE.search(text)
        return match.group(1).strip() if match else text.strip()
