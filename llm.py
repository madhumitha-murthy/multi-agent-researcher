# llm.py - Gemini wrapper with rate limiting + FakeModel for tests

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from config import GEMINI_MODEL, GEMINI_RATE_DELAY


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw: object = None


class ChatModel(Protocol):
    def call(self, system: str, messages: list[dict], tools: list | None = None) -> ModelResponse: ...


class GeminiModel:
    """Gemini with manual tool execution and rate limiting for free tier."""

    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._last_call = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < GEMINI_RATE_DELAY:
            time.sleep(GEMINI_RATE_DELAY - elapsed)
        self._last_call = time.time()

    def call(self, system: str, messages: list[dict], tools: list | None = None) -> ModelResponse:
        from google.genai import types

        self._rate_limit()

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

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

        resp = self.client.models.generate_content(
            model=self.model,
            contents=self._build_contents(messages),
            config=config,
        )

        content = resp.candidates[0].content
        calls = [
            {"name": fc.name, "args": dict(fc.args or {})}
            for fc in (resp.function_calls or [])
        ]
        text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None))
        return ModelResponse(text=text.strip(), tool_calls=calls, raw=content)

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

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, system: str, messages: list[dict], tools: list | None = None) -> ModelResponse:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.responses.pop(0) if self.responses else ModelResponse()
