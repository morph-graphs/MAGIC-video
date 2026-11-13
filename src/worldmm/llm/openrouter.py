"""
OpenRouter Model Wrapper
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
import logging
import copy
import base64
import io

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


def _repair_truncated_json(text: str) -> Any:
    """EPIC adapter: parse a JSON answer cut off by max_tokens by closing it after its last complete element."""
    ends = [i for i, ch in enumerate(text) if ch in "]}"]
    for end in reversed(ends[-400:]):
        prefix, stack, in_str, esc = text[: end + 1], [], False, False
        for ch in prefix:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}" and stack:
                stack.pop()
        if in_str:
            continue
        try:
            return json.loads(prefix + "".join("]" if c == "[" else "}" for c in reversed(stack)))
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("unrepairable truncated JSON", text, 0)


class OpenRouterModel:
    """
    OpenRouter model wrapper that provides the same interface as OpenAIModel.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenRouter model wrapper.

        Args:
            model_name: Name of the model to use (e.g., "openai/gpt-4o-mini")
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments
        """
        # EPIC adapter: MAGIC_LLM_BASE_URL points every LLM call (retriever, responder, time parser,
        # preprocessing) at the local server (epic/hf_server.py) serving the backbone as MAGIC_LLM_MODEL; no paid API.
        local = os.environ.get("MAGIC_LLM_BASE_URL")
        if local:
            self.client = OpenAI(api_key="EMPTY", base_url=local, timeout=1800)
        else:
            self.client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

        self.model_name = model_name
        self.max_retries = max(1, max_retries)
        self.kwargs = kwargs

    @staticmethod
    def _extract_json_block(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return text

    def _parse_structured_response(self, response_text: str, text_format: type) -> Any:
        cleaned = self._extract_json_block(response_text)
        try:
            json_data = json.loads(cleaned)
        except json.JSONDecodeError:
            if not os.environ.get("MAGIC_LLM_BASE_URL"):
                raise
            json_data = _repair_truncated_json(cleaned)
        # EPIC adapter, lenient parsing for local backbones: a bare JSON list for a schema with one
        # field is that field (the NER prompt literally asks for "a JSON list"), and triples with a
        # null element are dropped instead of failing the whole passage.
        fields = list(getattr(text_format, "model_fields", {}))
        if isinstance(json_data, list) and len(fields) == 1:
            json_data = {fields[0]: json_data}
        if isinstance(json_data, dict) and isinstance(json_data.get("triples"), list):
            json_data["triples"] = [t for t in json_data["triples"]
                                    if isinstance(t, list) and all(isinstance(x, str) for x in t)]
        if isinstance(json_data, dict) and isinstance(json_data.get("semantic_triples"), list):
            # keep (triple, evidence) pairs whose triple is 3 strings; evidence missing after a cut-off -> []
            ev = json_data.get("episodic_evidence") or []
            ev = ev + [[]] * (len(json_data["semantic_triples"]) - len(ev))
            pairs = [(t, [i for i in e if isinstance(i, int)] if isinstance(e, list) else [])
                     for t, e in zip(json_data["semantic_triples"], ev)
                     if isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t)]
            json_data["semantic_triples"] = [t for t, _ in pairs]
            json_data["episodic_evidence"] = [e for _, e in pairs]

        if hasattr(text_format, "model_validate"):
            return text_format.model_validate(json_data)
        if hasattr(text_format, "parse_obj"):
            return text_format.parse_obj(json_data)

        raise ValueError("text_format must be a Pydantic model class")

    @staticmethod
    def _sanitize_api_kwargs(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove local-only kwargs that are not valid OpenRouter Chat Completions params.
        """
        blocked_keys = {
            "fps",
            "nframes",
            "max_size",
            "max_size_video",
            "quality",
            "cache_dir",
        }
        dropped = blocked_keys.intersection(api_kwargs.keys())
        if dropped:
            logger.warning(
                "Dropping unsupported OpenRouter kwargs: %s",
                ", ".join(sorted(dropped)),
            )
        return {k: v for k, v in api_kwargs.items() if k not in blocked_keys}

    @staticmethod
    def _encode_image_to_data_url(image: Any) -> str:
        """Convert a PIL image (or image path) to a JPEG data URL for OpenRouter."""
        pil_image: Optional[Image.Image] = None
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            pil_image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image payload type: {type(image)}")

        buffer = io.BytesIO()
        rgb_image = pil_image.convert("RGB")
        rgb_image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize multimodal content to OpenRouter-compatible chat content format.
        """
        normalized = copy.deepcopy(messages)
        for message in normalized:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)  # type: ignore[arg-type]
                    continue

                item_type = item.get("type")
                if item_type == "text" and "text" in item:
                    new_content.append({"type": "text", "text": item["text"]})
                elif item_type == "image" and "image" in item:
                    image_url = self._encode_image_to_data_url(item["image"])
                    new_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                elif item_type == "image_url" and "image_url" in item:
                    new_content.append(item)
                else:
                    new_content.append(item)

            message["content"] = new_content

        return normalized

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Any:
        """
        Generate completion for a single prompt.

        Args:
            prompt: Conversation prompt (list of role/content dicts or raw string)

        Returns:
            Generated response string
        """
        # Normalize prompt to messages format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        messages = self._preprocess_messages(messages)

        text_format = kwargs.pop("text_format", None)

        merged_kwargs = self._sanitize_api_kwargs({**self.kwargs, **kwargs})

        # Move non-standard OpenAI params to extra_body for OpenRouter
        _extra_body_keys = {"top_k", "min_p", "repetition_penalty"}
        extra_body = {k: merged_kwargs.pop(k) for k in _extra_body_keys if k in merged_kwargs}
        if extra_body:
            merged_kwargs["extra_body"] = extra_body

        if os.environ.get("MAGIC_LLM_BASE_URL"):
            # local backbone: non-thinking mode (Qwen3.8), seeded sampling, bounded output length
            eb = merged_kwargs.setdefault("extra_body", {})
            eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
            merged_kwargs.setdefault("seed", int(os.environ.get("MAGIC_SEED", "0")))
            merged_kwargs.setdefault("max_tokens", int(os.environ.get("MAGIC_MAX_TOKENS", "4096")))
        response = self.client.chat.completions.create(
            model=os.environ.get("MAGIC_LLM_MODEL", self.model_name),
            messages=messages,
            **merged_kwargs,
        )

        if not response.choices:
            logger.warning("OpenRouter returned empty choices (model may not support this input type)")
            return None

        content = response.choices[0].message.content
        if text_format is None:
            return content

        try:
            return self._parse_structured_response(content, text_format)
        except Exception as e:
            logger.warning(f"Failed to parse structured response with {text_format}: {e}")
            raise

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], **kwargs) -> List[Any]:
        """
        Process multiple prompts in batch.

        Args:
            batch_prompts: List of conversation prompts

        Returns:
            List of response strings
        """
        results = []
        for prompt in batch_prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
