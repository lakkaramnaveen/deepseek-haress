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
    "plus a manifest mapping every file in the project from its original "
    "path to its new path. Rules:\n"
    "1. Preserve the program's exact behavior, structure, and public "
    "   names (functions, classes, exported symbols) as closely as "
    "   idiomatic {target} allows.\n"
    "2. Keep comments, translating their language but not their meaning.\n"
    "3. Rewrite import/require paths using the manifest's new paths "
    "   (matching their new extensions and any target-language "
    "   conventions), not the original {source} paths -- the manifest "
    "   covers every file in the project, including ones that import "
    "   *this* file, so use it even for forward references.\n"
    "4. Do not invent functionality that was not in the original file.\n"
    "5. Respond with a single fenced code block containing only the "
    "   complete translated file contents -- no explanations before or "
    "   after the code block, and no partial/truncated output."
)

# Matches a fenced code block: an opening ``` (optionally followed by a
# language tag) through to the LAST closing ``` in the response. Greedy on
# purpose: a translated whole file can itself legitimately contain a
# ``` example inside a comment or docstring, and a non-greedy match would
# stop at that *inner* fence and silently truncate everything after it.
# Since we always ask for a single fenced block with nothing else around
# it, capturing through to the final ``` is the correct reading of "give
# me everything you meant as the code block".
CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*)```", re.DOTALL)


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
        return self._complete(SYSTEM_PROMPT, user_prompt, temperature)

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
        is the project's full original-path -> new-path mapping (every
        file, not just ones already translated), so imports -- including
        forward references to files not yet translated -- stay consistent
        across the whole project.
        """
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(
            source=source_language, target=target_language
        )
        user_parts = [f"File: {rel_path}"]
        if manifest:
            user_parts.append(f"Project file manifest (original -> new):\n{manifest}")
        user_parts.append(f"Source ({source_language}):\n```\n{source_code}\n```")
        user_prompt = "\n\n".join(user_parts)

        return self._complete(system_prompt, user_prompt, temperature)

    def _complete(self, system_prompt: str, user_prompt: str, temperature: float) -> dict:
        """Shared chat-completion call used by generate_code and
        translate_file: send one system/user turn, extract the fenced
        code block from the reply, and return both alongside token usage."""
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
