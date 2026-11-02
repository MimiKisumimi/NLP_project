"""
Robot agent and LLM provider interface (`agent.py`).

Implements the `LLMProvider` / `GeminiProvider` structured-output client,
the autonomous `RobotAgent` decision loop, and the generalist vs. heterogeneous
robot capability profiles (`setskillset1`, `get_reference_profile`, `get_agent_profile`).
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Type

# Optional Google GenAI SDK imports (used if google-generativeai / google-genai is installed)
try:
    import google.generativeai as genai  # type: ignore
    import google.ai.generativelanguage as glm  # type: ignore

    HAS_GOOGLE_GENAI_SDK = True
except ImportError:
    try:
        from google import genai  # type: ignore
        from google.genai import types as glm  # type: ignore

        HAS_GOOGLE_GENAI_SDK = True
    except ImportError:
        genai = None
        glm = None
        HAS_GOOGLE_GENAI_SDK = False

from models import (
    ActionType,
    CommunicationIntent,
    LimitationCondition,
    Item,
    Message,
    RobotAction,
)
from config import LimitationConfig, get_condition_config


def load_env(dotenv_path: Optional[str] = None) -> bool:
    """
    Loads environment variables from a .env file.
    Supports `python-dotenv` if installed, with a built-in lightweight fallback parser.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        return load_dotenv(dotenv_path=dotenv_path or ".env", override=False)
    except ImportError:
        pass

    candidate_paths = [
        dotenv_path,
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    for p in candidate_paths:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if (v.startswith('"') and v.endswith('"')) or (
                                v.startswith("'") and v.endswith("'")
                            ):
                                v = v[1:-1]
                            if k and k not in os.environ:
                                os.environ[k] = v
                return True
            except Exception:
                pass
    return False


# Automatically load .env file on module import
load_env()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extracts and parses a JSON object from raw LLM text responses.
    Handles reasoning models that emit `<think>...</think>` blocks, Markdown
    fenced code blocks (` ```json ... ``` `), or leading/trailing prose around
    the outermost `{ ... }` JSON object.
    """
    if not text:
        return {}
    clean = text.strip()
    # Strip <think>...</think> reasoning blocks if present
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    # Strip markdown code blocks
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL)
        if match:
            clean = match.group(1).strip()
        else:
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        return json.loads(clean)
    except Exception:
        first = clean.find("{")
        last = clean.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(clean[first : last + 1])
        raise


# ---------------------------------------------------------
# Unified LLM Provider Interface
# ---------------------------------------------------------
class LLMProvider(ABC):
    """
    Abstract base interface for LLM backends (such as `GeminiProvider`)
    with built-in telemetry tracking for API call count, latency, token usage, and estimated USD cost.
    """

    # Pricing rates per 1,000 tokens (USD) for cost estimation
    PROMPT_COST_PER_1K: float = (
        0.0004048  # Based on an average cost of whole prompt execution of Gemini 3.5-flash-lite with output/input token ratio of 1/20
    )
    COMPLETION_COST_PER_1K: float = 0.00030

    def __init__(self):
        """Initializes telemetry counters and default decoding parameters (`temperature`, `num_ctx`, `num_predict`)."""
        self.call_count: int = 0
        self.total_latency_ms: float = 0.0
        self.latencies_ms: List[float] = []
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.decoding_params: Dict[str, Any] = {
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 1024,
        }

    def record_telemetry(
        self,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_rate_prompt_per_1k: Optional[float] = None,
        cost_rate_completion_per_1k: Optional[float] = None,
    ):
        """
        Records a single LLM API call's wall-clock latency (in ms), prompt token count,
        completion token count, and incremental estimated financial cost in USD.
        """
        self.call_count += 1
        self.total_latency_ms += max(0.0, float(latency_ms))
        self.latencies_ms.append(round(max(0.0, float(latency_ms)), 3))
        p_tok = max(0, int(prompt_tokens))
        c_tok = max(0, int(completion_tokens))
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += p_tok + c_tok
        p_rate = (
            cost_rate_prompt_per_1k
            if cost_rate_prompt_per_1k is not None
            else self.PROMPT_COST_PER_1K
        )
        c_rate = (
            cost_rate_completion_per_1k
            if cost_rate_completion_per_1k is not None
            else self.COMPLETION_COST_PER_1K
        )
        call_cost = (p_tok / 1000.0) * p_rate + (c_tok / 1000.0) * c_rate
        self.estimated_cost_usd += call_cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Returns accumulated API latency, token counts, decoding parameters, and
        estimated USD cost for this provider instance (consumed by `PassiveObserver`).
        """
        mean_lat = (
            (self.total_latency_ms / self.call_count) if self.call_count > 0 else 0.0
        )
        return {
            "call_count": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "decoding_params": dict(self.decoding_params),
        }

    @abstractmethod
    def generate_action(
        self,
        system_instruction: str,
        user_prompt: str,
        schema: Type[RobotAction] = RobotAction,
    ) -> Optional[RobotAction]:
        """Abstract method: queries the LLM and returns a validated `RobotAction` instance or `None`."""
        pass


# ---------------------------------------------------------
# Google Gemini API Provider Implementation
# ---------------------------------------------------------
def pydantic_to_openapi_3_schema(
    schema_input: Union[Type[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Converts a Pydantic model class or JSON schema dictionary into an OpenAPI 3.0
    schema conforming to Google Gemini's `responseSchema` format (inlining `$defs`/`$ref`
    and stripping unsupported keys such as `title`, `default`, `anyOf`, and `additionalProperties`).
    """
    if hasattr(schema_input, "model_json_schema"):
        raw_schema = schema_input.model_json_schema()
    elif hasattr(schema_input, "schema"):
        raw_schema = schema_input.schema()
    elif isinstance(schema_input, dict):
        raw_schema = copy.deepcopy(schema_input)
    else:
        return {}

    schema_copy = copy.deepcopy(raw_schema)
    defs = schema_copy.pop("$defs", None) or schema_copy.pop("definitions", None) or {}
    schema_copy.pop("$schema", None)

    def _resolve_refs(node: Any, seen: Optional[set] = None) -> Any:
        if seen is None:
            seen = set()
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = str(node["$ref"])
                ref_key = ref_path.split("/")[-1]
                if ref_key in defs and ref_path not in seen:
                    target_def = copy.deepcopy(defs[ref_key])
                    extra = {k: v for k, v in node.items() if k != "$ref"}
                    resolved = _resolve_refs(target_def, seen | {ref_path})
                    if isinstance(resolved, dict):
                        resolved.update(_resolve_refs(extra, seen))
                        return resolved
                    return resolved
            return {k: _resolve_refs(v, seen) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve_refs(item, seen) for item in node]
        return node

    inlined = _resolve_refs(schema_copy)

    ALLOWED_OPENAPI_KEYS = {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "properties",
        "required",
        "items",
    }

    def _clean_node(node: Any) -> Any:
        if not isinstance(node, dict):
            if isinstance(node, list):
                return [_clean_node(item) for item in node]
            return node

        curr = copy.deepcopy(node)
        is_nullable = bool(curr.get("nullable", False))

        # 1. Resolve anyOf / oneOf (Pydantic represents Optional[T] as anyOf: [T, {type: null}])
        any_of = curr.pop("anyOf", None) or curr.pop("oneOf", None)
        if any_of and isinstance(any_of, list):
            non_null = []
            for branch in any_of:
                if isinstance(branch, dict):
                    if branch.get("type") == "null":
                        is_nullable = True
                    else:
                        non_null.append(branch)
            if non_null:
                merged_branch = _clean_node(non_null[0])
                if isinstance(merged_branch, dict):
                    for k, v in merged_branch.items():
                        if k not in curr or k in (
                            "type",
                            "enum",
                            "items",
                            "properties",
                        ):
                            curr[k] = v
            if is_nullable:
                curr["nullable"] = True

        # 2. Resolve allOf
        all_of = curr.pop("allOf", None)
        if all_of and isinstance(all_of, list):
            for branch in all_of:
                merged_branch = _clean_node(branch)
                if isinstance(merged_branch, dict):
                    for k, v in merged_branch.items():
                        if k not in curr or k in (
                            "type",
                            "enum",
                            "items",
                            "properties",
                        ):
                            curr[k] = v

        # 3. Normalize type (e.g. OpenAPI 3.1 list ["string", "null"])
        curr_type = curr.get("type")
        if isinstance(curr_type, list):
            if "null" in curr_type:
                curr["nullable"] = True
                remaining = [t for t in curr_type if t != "null"]
                curr_type = remaining[0] if remaining else "string"
            else:
                curr_type = curr_type[0] if curr_type else "string"
            curr["type"] = curr_type

        # Infer type if absent
        if "type" not in curr:
            if "properties" in curr:
                curr["type"] = "object"
            elif "items" in curr:
                curr["type"] = "array"
            elif "enum" in curr:
                curr["type"] = "string"
            else:
                curr["type"] = (
                    "object" if isinstance(curr.get("properties"), dict) else "string"
                )

        if isinstance(curr.get("type"), str):
            curr["type"] = curr["type"].lower()

        # 4. Clean object properties recursively
        if (
            curr.get("type") == "object"
            and "properties" in curr
            and isinstance(curr["properties"], dict)
        ):
            curr["properties"] = {
                k: _clean_node(v) for k, v in curr["properties"].items()
            }

        # 5. Clean array items recursively
        if curr.get("type") == "array":
            if "items" in curr:
                curr["items"] = _clean_node(curr["items"])
            else:
                curr["items"] = {"type": "string"}

        # 6. Clean enum values (must be list of strings)
        if "enum" in curr and isinstance(curr["enum"], list):
            curr["enum"] = [str(x) for x in curr["enum"]]
            curr["type"] = "string"

        # 7. Clean required list (must contain only existing properties)
        if "required" in curr:
            if isinstance(curr["required"], list):
                if curr.get("type") == "object" and "properties" in curr:
                    curr["required"] = [
                        r for r in curr["required"] if r in curr["properties"]
                    ]
                if not curr["required"]:
                    curr.pop("required", None)
            else:
                curr.pop("required", None)

        # 8. Set nullable boolean flag
        if is_nullable or curr.get("nullable"):
            curr["nullable"] = True
        else:
            curr.pop("nullable", None)

        # 9. Retain strictly allowed Gemini OpenAPI 3.0 schema keys (strip title, default, additionalProperties)
        cleaned = {k: v for k, v in curr.items() if k in ALLOWED_OPENAPI_KEYS}
        return cleaned

    return _clean_node(inlined)


class GeminiProvider(LLMProvider):
    """
    Concrete LLM provider communicating with the Google Gemini API.
    Reads `GEMINI_API_KEY` from environment variables (`os.getenv("GEMINI_API_KEY")`)
    and enforces structured outputs using Gemini's JSON mode (responseMimeType='application/json'
    and OpenAPI 3.0 responseSchema).
    """

    COMPLETION_COST_PER_1K: float = 0.00030

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        temperature: float = 0.2,
        use_sdk: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initializes the `GeminiProvider` with API key, model name, and decoding parameters."""
        super().__init__()
        if not api_key:
            load_env()
        self.api_key: str = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        chosen_model = model or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        self.model: str = chosen_model
        self.timeout: float = timeout
        self.temperature: float = temperature
        self.use_sdk: bool = (
            use_sdk
            if use_sdk is not None
            else os.getenv("USE_GEMINI_SDK", "false").lower() in ("true", "1", "yes")
        )
        self.decoding_params = {
            "temperature": self.temperature,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

    def is_available(self) -> bool:
        """Checks if the GEMINI_API_KEY environment variable or argument is configured."""
        return bool(self.api_key)

    @classmethod
    def get_available_models(cls, api_key: Optional[str] = None) -> List[str]:
        """Returns known supported Google Gemini models."""
        return [
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

    def generate_structured_output(
        self,
        system_instruction: str,
        user_prompt: str,
        schema: Type[Any],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Queries Gemini with JSON mode (`responseMimeType='application/json'`) and an OpenAPI 3.0 `responseSchema`."""
        if not self.api_key:
            print(
                "[GeminiProvider] Error: GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable."
            )
            return None

        t_start = time.perf_counter()
        target_temp = temperature if temperature is not None else self.temperature

        # Convert Pydantic model / schema to Gemini-compliant OpenAPI 3.0 format
        openapi_schema = pydantic_to_openapi_3_schema(schema)

        # ---------------------------------------------------------------------
        # Branch 1: Official Google GenAI SDK (if requested & SDK is installed)
        # ---------------------------------------------------------------------
        if (
            self.use_sdk
            and HAS_GOOGLE_GENAI_SDK
            and glm is not None
            and genai is not None
        ):
            try:
                genai.configure(api_key=self.api_key)
                try:
                    generation_config = glm.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=target_temp,
                    )
                except Exception:
                    generation_config = glm.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=openapi_schema,
                        temperature=target_temp,
                    )

                sdk_model_clean = self.model
                if sdk_model_clean.startswith("models/"):
                    sdk_model_clean = sdk_model_clean[len("models/") :]

                model_inst = genai.GenerativeModel(
                    model_name=sdk_model_clean,
                    system_instruction=system_instruction,
                )
                try:
                    response = model_inst.generate_content(
                        user_prompt,
                        generation_config=generation_config,
                        request_options={"timeout": self.timeout},
                    )
                except (TypeError, ValueError):
                    response = model_inst.generate_content(
                        user_prompt,
                        generation_config=generation_config,
                    )

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                response_text = response.text if hasattr(response, "text") else ""

                p_tok = max(
                    1, len((system_instruction + " " + user_prompt).split()) * 4 // 3
                )
                c_tok = max(1, len(response_text.split()) * 4 // 3)
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    p_tok = getattr(usage_meta, "prompt_token_count", p_tok)
                    c_tok = getattr(usage_meta, "candidates_token_count", c_tok)

                self.record_telemetry(
                    latency_ms=elapsed_ms,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                )

                parsed_data = extract_json_from_text(response_text)
                if isinstance(schema, type) and (
                    hasattr(schema, "__fields__") or hasattr(schema, "model_fields")
                ):
                    return schema(**parsed_data)
                elif callable(schema):
                    return schema(**parsed_data)
                return parsed_data
            except Exception as sdk_err:
                print(
                    f"[GeminiProvider] Google GenAI SDK call failed: {sdk_err}. Falling back to REST API."
                )

        # ---------------------------------------------------------------------
        # Branch 2: REST API with explicit generationConfig (responseMimeType & responseSchema)
        # ---------------------------------------------------------------------
        model_clean = self.model
        if model_clean.startswith("models/"):
            model_clean = model_clean[len("models/") :]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": target_temp,
                "responseMimeType": "application/json",
                "responseSchema": openapi_schema,
            },
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                result = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            err_body = ""
            try:
                err_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            print(
                f"[GeminiProvider] HTTP {http_err.code} Error: {http_err.reason}\nDetails: {err_body}"
            )
            return None
        except urllib.error.URLError as url_err:
            print(f"[GeminiProvider] Network/URL Error: {url_err.reason}")
            return None
        except Exception as e:
            print(f"[GeminiProvider] Request to Gemini API failed: {e}")
            return None

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        candidates = result.get("candidates", [])
        if not candidates:
            print(f"[GeminiProvider] No candidates returned from Gemini API: {result}")
            return None

        response_text = ""
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                response_text += part["text"]

        if not response_text:
            print(f"[GeminiProvider] Empty text returned in candidate: {candidates[0]}")
            return None

        usage = result.get("usageMetadata", {})
        p_tok = int(
            usage.get("promptTokenCount")
            or max(1, len((system_instruction + " " + user_prompt).split()) * 4 // 3)
        )
        c_tok = int(
            usage.get("candidatesTokenCount")
            or max(1, len(response_text.split()) * 4 // 3)
        )

        self.record_telemetry(
            latency_ms=elapsed_ms,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )

        try:
            parsed_data = extract_json_from_text(response_text)
            if isinstance(schema, type) and (
                hasattr(schema, "__fields__") or hasattr(schema, "model_fields")
            ):
                return schema(**parsed_data)
            elif callable(schema):
                return schema(**parsed_data)
            return parsed_data
        except Exception as parse_err:
            print(
                f"[GeminiProvider] Failed to parse structured output ({parse_err}). Raw output:\n{response_text[:200]}"
            )
            return None

    def generate_action(
        self,
        system_instruction: str,
        user_prompt: str,
        schema: Type[RobotAction] = RobotAction,
    ) -> Optional[RobotAction]:
        """Generates a validated `RobotAction` for a robot agent turn."""
        return self.generate_structured_output(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            schema=schema,
        )


# ---------------------------------------------------------
# Robot Agent with Heterogeneous Capability Profile
# ---------------------------------------------------------
class RobotAgent:
    """
    Represents an autonomous robot agent operating in the state-based kitchen environment.
    Manages its location, physical skills, carrying capacity, tool compatibility, inventory,
    sliding-window conversation memory, communication budgets, and LLM decision loop.
    """

    def __init__(
        self,
        robot_id: str,
        skills: Optional[List[Union[ActionType, str]]] = None,
        carrying_capacity: int = 1,
        tool_compatibility: Optional[List[str]] = None,
        role: str = "General",
        description: str = "",
        llm_provider: Optional[LLMProvider] = None,
        memory_limit: Optional[int] = None,
        max_messages: Optional[int] = None,
        max_comm_tokens: Optional[int] = None,
        location: str = "Refrigerator",
    ):
        """
        Initializes a `RobotAgent` with its physical capability profile, experimental
        limitation constraints (`memory_limit`, `max_messages`, `max_comm_tokens`),
        and `LLMProvider`.
        """
        self.robot_id = robot_id
        self.location: str = location
        self.skills: List[str] = [
            s.value if isinstance(s, ActionType) else str(s) for s in (skills or [])
        ]
        self.carrying_capacity: int = carrying_capacity
        self.tool_compatibility: List[str] = (
            tool_compatibility if tool_compatibility is not None else ["all"]
        )
        self.role: str = role
        self.role_description: str = description

        # Limitation parameters
        self.memory_limit: Optional[int] = memory_limit
        self.max_messages: Optional[int] = max_messages
        self.max_comm_tokens: Optional[int] = max_comm_tokens
        self.messages_sent_count: int = 0
        self.comm_tokens_used: int = 0

        # Inventory
        self.inventory: List[Item] = []

        self.memory: List[Message] = []
        self.inbox: List[Message] = []
        self.local_observation: Dict[str, Any] = {}

        # Unified LLM provider interface
        self.llm_provider: LLMProvider = llm_provider or GeminiProvider()

        # Action duration, busy tracking, and previous action history
        self.busy_until: int = 0
        self.current_action: Optional[Dict[str, Any]] = None
        self.last_action_str: str = "NONE"

    def can_carry(self) -> bool:
        """Returns `True` if the robot currently holds fewer items than its `carrying_capacity`."""
        return len(self.inventory) < self.carrying_capacity

    def _find_item_index(self, item_name_or_id: str) -> Optional[int]:
        """
        Searches `self.inventory` for an item whose `name` or `id` matches `item_name_or_id`
        (case-insensitive) and returns its list index, or `None` if not held.
        """
        query = item_name_or_id.strip().lower()
        for idx, it in enumerate(self.inventory):
            if query in (it.name.lower(), it.id.lower()):
                return idx
        return None

    def has_item(self, item_name_or_id: str) -> bool:
        """Checks if the robot is currently holding an item matching `item_name_or_id`."""
        return self._find_item_index(item_name_or_id) is not None

    def get_item(self, item_name_or_id: str) -> Optional[Item]:
        """Retrieves an `Item` held in the robot's inventory by name or ID without removing it."""
        idx = self._find_item_index(item_name_or_id)
        return self.inventory[idx] if idx is not None else None

    def add_item(self, item: Item) -> bool:
        """Adds an `Item` to the robot's inventory if capacity permits and updates `item.location`."""
        if self.can_carry():
            self.inventory.append(item)
            item.location = self.robot_id
            return True
        return False

    def remove_item(self, item_name_or_id: str) -> Optional[Item]:
        """Removes and returns an `Item` from the robot's inventory by name or ID."""
        idx = self._find_item_index(item_name_or_id)
        return self.inventory.pop(idx) if idx is not None else None

    def is_busy(self, current_time: int) -> bool:
        """Returns `True` if the robot is still mid-execution on a multi-second physical action."""
        return current_time < self.busy_until

    def can_communicate(self, extra_tokens: int = 0) -> Tuple[bool, str]:
        """
        Checks whether sending a message with `extra_tokens` words is permitted under
        the agent's `max_messages` and `max_comm_tokens` budget (Conditions 4 & 5).
        Returns `(is_allowed, reason_if_blocked)`.
        """
        if (
            self.max_messages is not None
            and self.messages_sent_count >= self.max_messages
        ):
            return (
                False,
                f"Message count limit reached ({self.messages_sent_count}/{self.max_messages}).",
            )
        if (
            self.max_comm_tokens is not None
            and (self.comm_tokens_used + extra_tokens) > self.max_comm_tokens
        ):
            return (
                False,
                f"Communication token budget exceeded ({self.comm_tokens_used + extra_tokens}/{self.max_comm_tokens} tokens).",
            )
        return True, ""

    def _append_to_memory(self, msg: Message):
        """
        Appends a sent or received `Message` to `self.memory` and enforces the sliding-window
        memory cap (`self.memory_limit`) by retaining only the most recent `k` messages.
        """
        self.memory.append(msg)
        if self.memory_limit is not None and len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit :]

    def record_sent_message(self, msg: Message):
        """Records an outgoing message in the agent's memory and increments communication budget counters."""
        tokens = len(msg.message_content.split())
        self.messages_sent_count += 1
        self.comm_tokens_used += tokens
        self._append_to_memory(msg)

    def receive_message(self, msg: Message):
        """Called by `Environment` to deliver an incoming broadcast or direct message to this robot's inbox."""
        self.inbox.append(msg)
        self._append_to_memory(msg)
        intent_tag = f" [{msg.get_intent_str()}]" if msg.intent else ""
        print(
            f"[{self.robot_id}] Received message from {msg.sender}{intent_tag}: {msg.content}"
        )

    def update_observation(self, observation: Dict[str, Any]):
        """Called by `Environment` each timestep to refresh the robot's local observation dictionary."""
        self.local_observation = observation

    def get_visible_memory(self) -> List[Message]:
        """Returns the agent's message memory, truncated to the last `k` messages if memory-limited."""
        if self.memory_limit is not None and self.memory_limit > 0:
            return self.memory[-self.memory_limit :]
        return self.memory

    def _get_comm_budget_str(self) -> str:
        """Formats the remaining communication budget (messages and tokens left) for prompt awareness."""
        if self.max_messages is None and self.max_comm_tokens is None:
            return "Unlimited"
        rem_msgs = (
            f"{max(0, self.max_messages - self.messages_sent_count)}/{self.max_messages} msgs"
            if self.max_messages is not None
            else "unlimited msgs"
        )
        rem_toks = (
            f"{max(0, self.max_comm_tokens - self.comm_tokens_used)}/{self.max_comm_tokens} tokens"
            if self.max_comm_tokens is not None
            else "unlimited tokens"
        )
        return f"LIMITED ({rem_msgs} left, {rem_toks} left) - Be concise and only communicate when high-utility!"

    def _build_system_prompt(self) -> str:
        """
        Constructs the system instruction prompt for decentralized peer negotiation,
        informing the LLM of this robot's role, skills, capacity, tool compatibility,
        observability mode, communication budget, speech-act taxonomy, and single-turn rules.
        """
        obs_mode = self.local_observation.get("observability_mode", "full")
        comm_budget_str = self._get_comm_budget_str()
        return (
            f"You are {self.robot_id} ({self.role}) in a decentralized peer-to-peer multi-robot kitchen (NO central supervisor).\n"
            f"Your Physical Capabilities -> Skills: {', '.join(self.skills)} | Carrying Capacity: {self.carrying_capacity} | Compatible Tools: {', '.join(self.tool_compatibility)}.\n"
            f"Observability Mode: {obs_mode.upper()} | Communication Budget: {comm_budget_str}.\n\n"
            f"AGENT DECISION CYCLE (STRICTLY COMMUNICATE OR ACT):\n"
            f"Observe -> reason -> communicate OR act -> receive feedback -> update plan.\n"
            f"In each timestep, you must choose EITHER to send a message (`function='MESSAGE'`) OR to execute a single physical action (`MOVE`, `PICK`, `DROP`, `CUT`, `COOK`, `BOIL`, `WASH`, `ASSEMBLE`, `WAIT`). You cannot do both in the same turn.\n\n"
            f"DECENTRALIZED PEER NEGOTIATION RULES:\n"
            f"1. Discover & Announce Capabilities: No global allocation plan is provided. Coordinate with peers by sending messages (`function='MESSAGE'`) to announce capabilities (`intent='STATUS_UPDATE'`) or propose task divisions (`intent='PROPOSAL'`).\n"
            f"2. Speech Act Intents: Whenever `function='MESSAGE'`, classify your `intent` into one of the 8 speech acts:\n"
            f"   - PROPOSAL: Proposing a task division or sub-goal allocation.\n"
            f"   - COMMITMENT: Committing to perform a specific step (e.g. 'I will cut the tomato').\n"
            f"   - REQUEST: Asking a peer for tools, ingredient deliveries, or station access.\n"
            f"   - STATUS_UPDATE: Reporting capabilities, milestones, or station observations.\n"
            f"   - WARNING: Alerting peers about occupied stations or locked tools.\n"
            f"   - CONFLICT_RESOLUTION: Resolving disputes over shared resources (knife, stove, cutting table).\n"
            f"   - PLAN_REVISION: Modifying the plan after delays, blocks, or errors.\n"
            f"   - FAILURE_EXPLANATION: Explaining why an attempted action failed.\n"
            f"3. MANDATORY CONFLICT & FAILURE REACTION PROTOCOLS:\n"
            f"   - On Resource Contention: If a station or tool you need is occupied or locked by a peer, DO NOT repeatedly attempt the action and DO NOT silently WAIT. On your next turn, send a MESSAGE with `intent='CONFLICT_RESOLUTION'` or `intent='REQUEST'` to the occupying robot to coordinate who yields or when it will be released, OR switch to another ready subtask.\n"
            f"   - On Action Rejection / Failure: If your previous action was rejected or failed (e.g., resource locked, missing ingredient, skill mismatch), DO NOT blindly retry or stay silent. Send a MESSAGE with `intent='FAILURE_EXPLANATION'` informing your peers of the obstacle, or `intent='PLAN_REVISION'` to reassign who handles the delayed step.\n"
            f"   - Anti-Looping Rule: Never attempt the exact same failed action two timesteps in a row without communicating with your peers or moving to resolve the blocker.\n\n"
            f"CRITICAL PHYSICAL & NAVIGATION RULES:\n"
            f"1. To move to a station, set `function='MOVE'` and `target_station` to one of: 'Refrigerator', 'Cutting table', 'Sink', 'Stove', 'Food assembly table', 'Serving table'.\n"
            f"2. You must be located at the required workstation before performing station actions.\n"
            f"3. You can ONLY perform actions in your Skills. If you cannot cut or cook an item, MOVE to the required station, DROP the item, and on a communication turn send a REQUEST or STATUS_UPDATE to a capable peer.\n"
            f"4. If your inventory is FULL ({len(self.inventory)}/{self.carrying_capacity}), you must MOVE to a station and DROP items before picking more.\n"
            f"5. Exact Item Naming: When using PICK, DROP, CUT, COOK, BOIL, WASH, ASSEMBLE, set `target_item` to the exact base name (e.g. 'tomato', 'lettuce', 'meat_patty', 'salad') WITHOUT state labels like '(raw)'. When using DROP, you can set `target_item` to a specific item name or 'all' to unload your entire inventory at the workstation.\n\n"
            f"CULINARY DOMAIN RULES (DO NOT EXECUTE INVALID OPERATIONS):\n"
            f"- meat_patty: Must be COOKED on the Stove using a pan. NEVER CUT or WASH meat patties! (Cutting a meat patty is strictly forbidden and rejected).\n"
            f"- burger_bun and bread: Kept RAW. DO NOT CUT, COOK, BOIL, or WASH buns or bread. Transport directly to the Food assembly table.\n"
            f"- vegetables and fruits (tomato, lettuce, cucumber, cheese, carrot, potato, orange): CUT on the Cutting table using a knife.\n"
            f"- soup vegetables (carrot, potato): CUT at Cutting table, then BOILED at the Stove using a pot.\n"
            f"- washables (orange, veggies): Can be WASHED at the Sink.\n"
            f"- ASSEMBLE: Must be done at Food assembly table on a container (plate, bowl, cup) when all required ingredients are ready. `target_item` MUST be the final dish name (e.g., 'hamburger', 'salad', 'vegetable_soup')."
        )

    def _build_user_prompt(self, timestep: int) -> str:
        """
        Constructs the per-timestep user prompt containing the current task goal,
        active workstation/tool locks, robot locations,
        station inventories, unread inbox messages, and recent conversation history.
        """
        obs_mode = self.local_observation.get("observability_mode", "full")
        comm_budget_str = self._get_comm_budget_str()

        memory_lines = [
            f"T{m.timestep}s | {m.sender} -> {m.recipient} [{m.get_intent_str()}]: {m.content}"
            for m in self.get_visible_memory()
        ]
        memory_str = "\n".join(memory_lines) if memory_lines else "No past messages."

        inbox_lines = [
            f"T{m.timestep}s | {m.sender} -> {m.recipient} [{m.get_intent_str()}]: {m.content}"
            for m in self.inbox
        ]
        inbox_str = (
            "\n".join(inbox_lines)
            if inbox_lines
            else "No new unread messages this turn."
        )
        # Clear unread inbox after reading into prompt
        self.inbox.clear()

        my_loc = self.local_observation.get("my_location", self.location)
        other_locs = self.local_observation.get("other_robot_locations", {})

        inv_items = [f"{it.name} ({it.state})" for it in self.inventory]
        inv_str = ", ".join(inv_items) if inv_items else "Empty"

        station_items_info = self.local_observation.get("workstation_items", {})
        station_summary = "\n".join(
            [f"  - {st}: {items}" for st, items in station_items_info.items()]
        )

        task_desc = self.local_observation.get(
            "task_description", "Prepare the requested dish"
        )
        target_dishes = self.local_observation.get("target_dishes", ["dish"])
        target_station = self.local_observation.get("target_station", "Serving table")
        ws_locks = self.local_observation.get("workstation_locks", {})
        tool_locks = self.local_observation.get("tool_locks", {})

        locks_summary = []
        if ws_locks:
            locks_summary.append(f"Occupied Stations: {ws_locks}")
        if tool_locks:
            locks_summary.append(f"Locked Tools: {tool_locks}")
        locks_str = (
            " | ".join(locks_summary)
            if locks_summary
            else "All visible stations and tools available"
        )

        failure_alert = ""
        if (
            str(self.last_action_str).startswith("FAILED")
            or "rejected" in inbox_str.lower()
        ):
            failure_alert = (
                f"⚠️ ALERT: Your previous action FAILED or was REJECTED ({self.last_action_str})!\n"
                f"- DO NOT repeat the exact same action immediately, and DO NOT silently WAIT without communicating.\n"
                f"- If blocked by a locked station or tool, send a MESSAGE with `intent='CONFLICT_RESOLUTION'` or `intent='REQUEST'` to the occupying robot, or `function='MOVE'` to another workstation.\n"
                f"- If missing items or prerequisites, broadcast a MESSAGE with `intent='FAILURE_EXPLANATION'` or `intent='PLAN_REVISION'`.\n\n"
            )

        return (
            f"Current Time: {timestep}s | Goal: {task_desc} (Target: {', '.join(target_dishes)} at {target_station})\n"
            f"Locks: {locks_str}\n"
            f"{self.robot_id} ({self.role}) | Skills: {', '.join(self.skills)} | Capacity: {len(self.inventory)}/{self.carrying_capacity} | Inventory: [{inv_str}] | Previous Action: {self.last_action_str}\n"
            f"Current Location: {my_loc} | Visible Other Robots: {other_locs} | Comm Budget: {comm_budget_str}\n"
            f"Station Items ({obs_mode} view):\n{station_summary}\n"
            f"New Inbox Messages:\n{inbox_str}\n"
            f"Recent Conversation History (Memory):\n{memory_str}\n\n"
            f"{failure_alert}"
            f"Available Actions (`function` / `action` — choose ONE per turn: either communicate OR act):\n"
            f"1. MOVE: Move directly to a station by setting `target_station` ('Refrigerator', 'Cutting table', 'Sink', 'Stove', 'Food assembly table', 'Serving table').\n"
            f"2. PICK / DROP: At a station, pick up or drop an item by setting `target_item='item_name'`.\n"
            f"3. CUT / COOK / BOIL / WASH / ASSEMBLE: At the required station, process an item by setting `target_item`. REMINDER: meat_patty is COOKED (never cut), veggies are CUT, soup veggies are BOILED, and buns are RAW. For ASSEMBLE, `target_item` MUST be the name of the final dish (e.g., 'salad', 'hamburger'), NOT an ingredient.\n"
            f"4. MOVE_ITEMS: At the Refrigerator, fetch ingredients into inventory.\n"
            f"5. MESSAGE: Communicate with peers (`recipient='ALL'` or `'Robot_X'`, `intent='PROPOSAL'|'COMMITMENT'|'REQUEST'|'STATUS_UPDATE'|...`, `message_content='...'`, you MUST specify 'intent' from the 8 speech acts, and 'message_content').\n"
            f"6. WAIT: Idle for 1s.\n\n"
            f"CRITICAL: Output valid JSON matching the schema.\n"
            f'Example physical action: {{"should_act": true, "function": "MOVE", "target_station": "Cutting table", "content": "Moving to Cutting table."}}\n'
            f'Example conflict resolution: {{"should_act": true, "function": "MESSAGE", "recipient": "Robot_1", "intent": "CONFLICT_RESOLUTION", "message_content": "Robot_1, please notify me when you finish cutting so I can use the Cutting table."}}\n'
            f'Example failure explanation: {{"should_act": true, "function": "MESSAGE", "recipient": "ALL", "intent": "FAILURE_EXPLANATION", "message_content": "Action rejected because Cutting table is locked; switching to Refrigerator."}}\n\n'
            f"What is your next decision?"
        )

    def reason_and_act(self, timestep: int) -> Optional[Message]:
        """
        Queries the unified `LLMProvider` based on `local_observation` and `memory` to decide
        the robot's next action at `timestep`.
        Strictly enforces the 'Observe -> reason -> communicate or act -> receive feedback -> update plan' cycle.
        Skips the LLM call and returns `None` if the robot is currently busy executing an action.
        """
        if self.is_busy(timestep):
            remaining = self.busy_until - timestep
            act_name = (
                self.current_action.get("function") if self.current_action else "action"
            )
            print(
                f"[{self.robot_id}] Busy performing {act_name} ({remaining}s remaining)..."
            )
            return None

        print(
            f"[{self.robot_id}] Reasoning based on observations and memory at t={timestep}s..."
        )

        system_instruction = self._build_system_prompt()
        user_prompt = self._build_user_prompt(timestep)

        action = self.llm_provider.generate_action(
            system_instruction, user_prompt, RobotAction
        )
        if action and action.should_act:
            act_func = action.function
            target_desc = action.target or "(none)"
            intent_label = f" | Intent: {action.intent.value}" if action.intent else ""
            msg_preview = (
                f' | Message: "{action.message_content}"'
                if act_func == ActionType.MESSAGE
                else ""
            )
            print(
                f"[{self.robot_id}] -> Selected {act_func.value} | Target: {target_desc}{intent_label}{msg_preview}"
            )

            # Snapshot compact local state for message analysis
            local_state_snapshot = {
                "location": self.local_observation.get("my_location", self.location),
                "inventory": [it.to_dict() for it in self.inventory],
                "workstation_items": dict(
                    self.local_observation.get("workstation_items", {})
                ),
                "other_robot_locations": dict(
                    self.local_observation.get("other_robot_locations", {})
                ),
                "observability_mode": self.local_observation.get(
                    "observability_mode", "full"
                ),
            }

            return Message(
                sender=self.robot_id,
                recipient=action.recipient,
                timestep=timestep,
                function=act_func,
                content=action.content,
                message_content=(
                    action.message_content if act_func == ActionType.MESSAGE else ""
                ),
                intent=action.intent if act_func == ActionType.MESSAGE else None,
                local_state=local_state_snapshot,
                previous_action=self.last_action_str,
                target_station=action.target_station,
                target_item=action.target_item,
                target=action.target,
            )
        elif action and not action.should_act:
            print(
                f"[{self.robot_id}] -> Chose to idle this timestep (should_act=False)."
            )
        else:
            print(f"[{self.robot_id}] -> No action received from provider.")
        return None


# ---------------------------------------------------------
# Robot Capability Profiles
# ---------------------------------------------------------
# Full skill set assigned to generalist robots under non-capability-limited conditions.
BROAD_SKILLS: List[ActionType] = [
    ActionType.PICK,
    ActionType.DROP,
    ActionType.CUT,
    ActionType.WASH,
    ActionType.COOK,
    ActionType.BOIL,
    ActionType.ASSEMBLE,
]


def setskillset1(robot_id: str) -> Dict[str, Any]:
    """
    Returns heterogeneous physical capability profile for a given robot (Condition 2 & 5).
    - Carrier (`Robot_1`): High carrying capacity (3), fetch/transport skills, no cut/cook.
    - Manipulation (`Robot_2`): Moderate capacity (1), dexterous cut/assemble/wash skills.
    - Heat-resistant (`Robot_3`): Moderate capacity (1), stove/cook/boil specialist.
    """
    profiles = {
        "Robot_1": {
            "role": "Carrier",
            "skills": [ActionType.PICK, ActionType.DROP],
            "carrying_capacity": 3,
            "tool_compatibility": ["all"],
            "description": "Carrier robot: high carrying capacity (3 items), picks and transports items, cannot cut or cook.",
        },
        "Robot_2": {
            "role": "Manipulation",
            "skills": [
                ActionType.CUT,
                ActionType.ASSEMBLE,
                ActionType.WASH,
                ActionType.PICK,
                ActionType.DROP,
            ],
            "carrying_capacity": 1,
            "tool_compatibility": ["knife", "plate", "bowl", "cup"],
            "description": "Manipulation robot: dexterous cutter, washer, and assembler (capacity 1).",
        },
        "Robot_3": {
            "role": "Heat-resistant",
            "skills": [
                ActionType.COOK,
                ActionType.BOIL,
                ActionType.PICK,
                ActionType.DROP,
            ],
            "carrying_capacity": 1,
            "tool_compatibility": ["pan", "pot"],
            "description": "Heat-resistant robot: specialized for stove cooking and boiling (capacity 1).",
        },
    }
    return profiles.get(
        robot_id,
        {
            "role": "General",
            "skills": [ActionType.PICK, ActionType.DROP],
            "carrying_capacity": 1,
            "tool_compatibility": ["all"],
            "description": "Standard general-purpose robot.",
        },
    )


def get_reference_profile(robot_id: str) -> Dict[str, Any]:
    """
    Returns broad, unconstrained capability profile for a robot (Reference, Observation-limited, Communication-limited).
    Every robot can perform all kitchen operations (`BROAD_SKILLS`) and carry up to 3 items.
    """
    return {
        "role": "Generalist",
        "skills": list(BROAD_SKILLS),
        "carrying_capacity": 3,
        "tool_compatibility": ["all"],
        "description": f"Broad-capability generalist robot ({robot_id}): can perform all kitchen tasks and carry up to 3 items.",
    }


def get_agent_profile(
    robot_id: str,
    condition: Union[
        LimitationCondition, LimitationConfig, str, int
    ] = LimitationCondition.CAPABILITY_LIMITED,
) -> Dict[str, Any]:
    """
    Returns the complete agent initialization profile (capabilities + limitation parameters)
    for a given `robot_id` under the specified `LimitationCondition`.
    Combines `setskillset1` or `get_reference_profile` with `memory_limit_k`,
    `max_messages_per_agent`, and `max_comm_tokens_per_agent`.
    """
    cond_cfg = get_condition_config(condition)
    base_profile = (
        setskillset1(robot_id)
        if cond_cfg.heterogeneous_capabilities
        else get_reference_profile(robot_id)
    )
    return {
        **base_profile,
        "memory_limit": cond_cfg.memory_limit_k,
        "max_messages": cond_cfg.max_messages_per_agent,
        "max_comm_tokens": cond_cfg.max_comm_tokens_per_agent,
    }
