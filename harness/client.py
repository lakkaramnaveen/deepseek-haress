"""Thin wrapper around DeepSeek's OpenAI-compatible chat completions API."""

import os
import re

from openai import OpenAI

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-coder",
        "signup_url": "https://platform.deepseek.com/api_keys",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "deepseek/deepseek-coder",
        "signup_url": "https://openrouter.ai/keys",
    },
}

SYSTEM_PROMPT = (
    "You are a careful coding assistant. Solve the given task in the "
    "requested language. Respond with a single fenced code block "
    "containing only the final, runnable solution -- no explanations "
    "before or after the code block."
)

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, provider: str | None = None):
        provider = provider or os.environ.get("DEEPSEEK_PROVIDER", "deepseek")
        if provider not in PROVIDERS:
            raise RuntimeError(
                f"Unknown provider {provider!r}. Set DEEPSEEK_PROVIDER to one "
                f"of: {', '.join(PROVIDERS)}"
            )
        config = PROVIDERS[provider]

        api_key = api_key or os.environ.get(config["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"{config['api_key_env']} is not set. Copy .env.example to "
                f".env and add your key from {config['signup_url']}"
            )
        self.provider = provider
        self.model = model or os.environ.get("DEEPSEEK_MODEL", config["default_model"])
        self.client = OpenAI(api_key=api_key, base_url=config["base_url"])

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
