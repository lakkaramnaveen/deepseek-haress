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

TRANSLATE_SYSTEM_PROMPT = (
    "You are an expert polyglot software engineer performing a "
    "whole-codebase source-to-source translation from {source} to "
    "{target}. You are shown one file at a time from a larger project, "
    "plus a manifest of sibling files already translated in this same "
    "run. Rules:\n"
    "1. Preserve the program's exact behavior, structure, and public "
    "   names (functions, classes, exported symbols) as closely as "
    "   idiomatic {target} allows.\n"
    "2. Keep comments, translating their language but not their meaning.\n"
    "3. Rewrite import/require paths to point at the already-translated "
    "   sibling files listed in the manifest (matching their new "
    "   extensions and any target-language conventions), not the "
    "   original {source} paths.\n"
    "4. Do not invent functionality that was not in the original file.\n"
    "5. Respond with a single fenced code block containing only the "
    "   complete translated file contents -- no explanations before or "
    "   after the code block, and no partial/truncated output."
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

    def translate_file(
        self,
        source_code: str,
        source_language: str,
        target_language: str,
        rel_path: str,
        manifest: str = "",
        temperature: float = 0.1,
    ) -> dict:
        """Ask the model to translate one file of a larger codebase.

        `rel_path` is the file's path relative to the project root (helps
        the model understand its role, e.g. `utils/format.py`). `manifest`
        is a short text listing sibling files already translated in this
        run (original path -> new path), so imports stay consistent
        across files.
        """
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(
            source=source_language, target=target_language
        )
        user_parts = [f"File: {rel_path}"]
        if manifest:
            user_parts.append(
                "Files already translated in this project "
                f"(original -> new):\n{manifest}"
            )
        user_parts.append(f"Source ({source_language}):\n```\n{source_code}\n```")
        user_prompt = "\n\n".join(user_parts)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
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
