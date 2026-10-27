#!/usr/bin/env python3
"""Minimal OpenRouter chat script.

Configuration comes from a .env file (see .env.example):
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=openai/gpt-4o-mini

Usage:
    python openrouter_chat.py "Explain what a hash table is"
    echo "Write a haiku about rust" | python openrouter_chat.py
"""

import json
import os
import ssl
import sys
import urllib.error
import urllib.request

import certifi
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get(
    "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def chat(prompt: str, model: str | None = None, system: str | None = None) -> str:
    model = model or MODEL
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({"model": model, "messages": messages}).encode()
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "openrouter-chat",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=60, context=SSL_CONTEXT
        ) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API error {exc.code}: {exc.read().decode()}")

    return data["choices"][0]["message"]["content"]


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("Provide a prompt as an argument or via stdin")

    print(chat(prompt))


if __name__ == "__main__":
    main()
