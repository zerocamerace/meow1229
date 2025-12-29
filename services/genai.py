from __future__ import annotations

import json
import logging
import re
import time

from google import genai
from google.genai import types as genai_types

from config.settings import GEMINI_API_KEY

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment")
    raise ValueError("GEMINI_API_KEY is required")

try:
    genai_chat_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as exc:
    logging.error("Failed to initialise google-genai client: %s", exc)
    raise

JSON_RESPONSE_CONFIG = genai_types.GenerateContentConfig(
    response_mime_type="application/json",
    candidate_count=1,
    temperature=0.6,
)

MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def extract_json_from_response(text: str) -> dict:
    if text is None:
        raise ValueError("LLM returned None")

    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in: {raw[:200]}")

    candidate = match.group(0)
    candidate = (
        candidate.replace("\u201d", '"')
        .replace("\u2019", "'")
        .replace("\ufeff", "")
    )

    return json.loads(candidate)


def build_genai_contents(system_instruction, conversation_history):
    contents = []

    if system_instruction:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=str(system_instruction))],
            )
        )

    for msg in conversation_history:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        if not parts:
            logging.warning("Empty parts in message: %s", msg)
            continue

        part_obj = parts[0]
        text = part_obj.get("text") if isinstance(part_obj, dict) else str(part_obj)
        if not text:
            logging.warning("Empty text in message: %s", msg)
            continue

        genai_role = "model" if role == "model" else "user"
        contents.append(
            genai_types.Content(
                role=genai_role,
                parts=[genai_types.Part(text=text)],
            )
        )

    return contents


def generate_with_retry(contents, generation_config=None):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        for attempt in range(3):
            try:
                kwargs = {"model": model_name, "contents": contents}
                if generation_config is not None:
                    kwargs["config"] = generation_config

                response = genai_chat_client.models.generate_content(**kwargs)
                if getattr(response, "candidates", None):
                    if attempt > 0 or model_name != MODEL_CANDIDATES[0]:
                        logging.warning(
                            "Gemini model '%s' succeeded on attempt %s",
                            model_name,
                            attempt + 1,
                        )
                    return response
                logging.warning(
                    "Gemini model '%s' returned empty candidates on attempt %s",
                    model_name,
                    attempt + 1,
                )
            except Exception as exc:
                logging.warning(
                    "Gemini model '%s' failed on attempt %s: %s",
                    model_name,
                    attempt + 1,
                    exc,
                )
                last_error = exc
            time.sleep(1)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini API returned empty response for all candidate models")
