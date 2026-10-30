# llm.py - Gemini wrapper with rate limiting + FakeModel for tests

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from config import GEMINI_MODEL, GEMINI_RATE_DELAY


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw: object = None
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    def call(self, system: str, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> ModelResponse: ...


def strip_json_fences(text: str) -> str:
    """Strip ```json ... ``` fences that Gemini often wraps around JSON."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class GeminiModel:
    """Gemini with manual tool execution and rate limiting."""

    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._last_call = 0.0
        self._lock = threading.Lock()

    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < GEMINI_RATE_DELAY:
                time.sleep(GEMINI_RATE_DELAY - elapsed)
            self._last_call = time.time()

    def call(self, system: str, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> ModelResponse:
        from google.genai import types

        self._rate_limit()

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        if json_mode:
            config.response_mime_type = "application/json"

        if tools:
            config.tools = [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters_json_schema=t["params"],
                    )
                    for t in tools
                ])
            ]

        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=self._build_contents(messages),
                    config=config,
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise

        content = resp.candidates[0].content
        calls = [
            {"name": fc.name, "args": dict(fc.args or {})}
            for fc in (resp.function_calls or [])
        ]
        text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None))

        # Extract token usage if available
        usage = getattr(resp, "usage_metadata", None)
        input_tok = getattr(usage, "prompt_token_count", 0) or 0
        output_tok = getattr(usage, "candidates_token_count", 0) or 0

        return ModelResponse(
            text=text.strip(), tool_calls=calls, raw=content,
            input_tokens=input_tok, output_tokens=output_tok,
        )

    @staticmethod
    def _build_contents(messages: list[dict]) -> list:
        from google.genai import types

        contents = []
        tool_results = []

        for m in messages:
            if m["role"] == "tool":
                tool_results.append(
                    types.Part.from_function_response(
                        name=m["name"], response={"result": m["result"]}
                    )
                )
                continue

            if tool_results:
                contents.append(types.Content(role="user", parts=list(tool_results)))
                tool_results.clear()

            if m["role"] == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=m["text"])])
                )
            elif m.get("raw") is not None:
                contents.append(m["raw"])
            else:
                parts = []
                if m.get("text"):
                    parts.append(types.Part.from_text(text=m["text"]))
                for c in m.get("tool_calls", []):
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=c["name"], args=c["args"])
                    ))
                contents.append(types.Content(role="model", parts=parts))

        if tool_results:
            contents.append(types.Content(role="user", parts=list(tool_results)))

        return contents


class FakeModel:
    """Scripted responses for tests. No API calls."""

    def __init__(self, responses: list[ModelResponse], fake_tokens: tuple[int, int] = (10, 5)):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._fake_tokens = fake_tokens  # (input, output) per call

    def call(self, system: str, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> ModelResponse:
        self.calls.append({"system": system, "messages": list(messages)})
        if self.responses:
            resp = self.responses.pop(0)
            # Apply fake token counts if not already set
            if resp.input_tokens == 0:
                resp.input_tokens = self._fake_tokens[0]
            if resp.output_tokens == 0:
                resp.output_tokens = self._fake_tokens[1]
            return resp
        return ModelResponse(input_tokens=self._fake_tokens[0], output_tokens=self._fake_tokens[1])
