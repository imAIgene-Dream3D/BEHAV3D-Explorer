"""
BEHAV3D assistant — a researcher's own model provider (no Qt).

With an own key the plugin calls the provider itself, so the key never reaches the
assistant service. One turn is three calls:

    POST {endpoint}/prepare   a canned reply, or the model request to send
    provider                  streamed with the researcher's key
    POST {endpoint}/finish    the reply, turned into validated tool calls

The providers are not interchangeable. Current OpenAI models only call tools through
the Responses API, and current Claude models reject ``temperature`` and a forced
``tool_choice``, so each adapter below speaks its provider's own API.

Keep the key out of everything that can be logged or shown: it travels as
:class:`ApiKey`, and every SDK error becomes a fixed message raised ``from None``,
because error text reaches the crash logs, the chat history and the next request.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

KEYRING_SERVICE = "BEHAV3D Assistant"
_PROVIDER_READ_TIMEOUT = 180.0   # thinking models can pause for a long time
_FINISH_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 2
_RETRYABLE_FAILURES = {"rate_limit", "server", "timeout"}
_ANTHROPIC_MAX_TOKENS = 64000
_ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Haiku 5.5 and older models have no server-side fallback.
_ANTHROPIC_FALLBACK_MODELS = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5-5")
_NON_CHAT_MODEL_WORDS = (
    "embed", "whisper", "tts", "dall-e", "moderation", "image", "realtime",
    "audio", "transcribe", "sora", "ocr",
)
_MUST_CALL_TOOL = (
    "\n\nThis request requires a change. Make it with the available tools in this "
    "reply. Do not describe a change without calling a tool."
)
_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Confirms that tools are available.",
        "parameters": {"type": "object", "properties": {}},
    },
}


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    adapter: str
    base_url: str
    temperature: Optional[float] = None
    # Whether the provider can be told a tool call is mandatory. Otherwise the
    # request says so in words, and /finish reports a reply that ignored it.
    forces_tool_calls: bool = False
    forced_tool_body: Optional[dict] = None
    default_model: str = ""


PROVIDERS = (
    ProviderSpec(
        "openai", "OpenAI", "openai_responses", "https://api.openai.com/v1",
    ),
    ProviderSpec(
        "anthropic", "Anthropic Claude", "anthropic", "https://api.anthropic.com",
        default_model="claude-opus-5-5",
    ),
    ProviderSpec(
        "gemini", "Google Gemini", "chat_completions",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
    ),
    ProviderSpec(
        "mistral", "Mistral", "chat_completions", "https://api.mistral.ai/v1",
        temperature=0.3, forces_tool_calls=True,
    ),
    ProviderSpec(
        "deepseek", "DeepSeek", "chat_completions", "https://api.deepseek.com",
        temperature=0.3, forces_tool_calls=True,
        # DeepSeek rejects a forced tool call while thinking is on.
        forced_tool_body={"thinking": {"type": "disabled"}},
    ),
)


def provider_by_id(provider_id: str) -> Optional[ProviderSpec]:
    return next((spec for spec in PROVIDERS if spec.id == provider_id), None)


class ApiKey:
    """A key that cannot end up in a log, a traceback's locals, or a chat message."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = str(value or "").strip()

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "ApiKey(<redacted>)"

    __str__ = __repr__


@dataclass
class ProviderConfig:
    spec: ProviderSpec
    model: str
    key: ApiKey


@dataclass
class ProviderReply:
    text: str = ""
    fragments: tuple = ()
    stop: str = "complete"          # complete | truncated | refused
    refusal_category: str = ""


class ProviderCancelled(Exception):
    """New chat or shutdown ended the turn."""


class ServiceTooOld(Exception):
    """The assistant service predates own-provider support."""


_FAILURE_TEXT = {
    "bad_key": "{label} rejected the API key. Check it in the assistant settings.",
    "forbidden": (
        "{label} refused this key. Check its permissions and billing in your {label} "
        "account."
    ),
    "no_model": (
        "{label} does not offer the model {model}. Choose another in the assistant "
        "settings."
    ),
    "rate_limit": (
        "{label} is rate limiting this key, or its quota has run out. Try again "
        "shortly, or check your {label} account."
    ),
    "timeout": "{label} did not respond in time.",
    "server": "{label} had a server problem. Try again shortly.",
    "connection": "Could not reach {label}. Check the internet connection.",
    "bad_request": (
        "{label} rejected the request. The model {model} may not support what the "
        "assistant needs."
    ),
    "missing_package": (
        "The {package} package is not installed in this environment. Reinstall "
        "BEHAV3D, or choose the BEHAV3D service in the assistant settings."
    ),
    "failed": "{label} could not generate a response.",
}


class ProviderError(Exception):
    """A provider failure with a fixed message, safe to show, log, and resend."""

    def __init__(self, kind: str, config: ProviderConfig, *, package: str = ""):
        self.kind = kind
        self.message = _FAILURE_TEXT[kind].format(
            label=config.spec.label, model=config.model, package=package,
        )
        super().__init__(self.message)

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE_FAILURES

    def status(self) -> dict:
        return {
            "level": "error",
            "stage": "provider",
            "component": "byok",
            # Not provider_*: those trigger a probe of the shared DeepSeek account.
            "code": f"byok_{self.kind}",
            "message": self.message,
            "server_error": True,
        }


def _failure_kind(error: Exception) -> str:
    name = type(error).__name__
    if name == "APITimeoutError":
        return "timeout"
    if name == "APIConnectionError":
        return "connection"
    status = getattr(error, "status_code", None)
    if status == 401:
        return "bad_key"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "no_model"
    if status == 429:
        return "rate_limit"
    if status in (400, 422):
        return "bad_request"
    if isinstance(status, int) and status >= 500:
        return "server"
    return "failed"


def _make_client(config: ProviderConfig):
    """Build an SDK client. Both arguments are explicit because the SDKs otherwise
    read ambient credentials and ``*_BASE_URL`` from the environment."""
    package = "anthropic" if config.spec.adapter == "anthropic" else "openai"
    try:
        if package == "anthropic":
            from anthropic import Anthropic as Client
        else:
            from openai import OpenAI as Client
    except ImportError:
        raise ProviderError("missing_package", config, package=package) from None
    return Client(
        api_key=config.key.reveal(),
        base_url=config.spec.base_url,
        timeout=_PROVIDER_READ_TIMEOUT,
        max_retries=0,
    )


def _system_text(spec: ProviderSpec, request: dict) -> str:
    system = str(request.get("system") or "")
    if request.get("must_call_tool") and not spec.forces_tool_calls:
        return system + _MUST_CALL_TOOL
    return system


def _chat_messages(request: dict) -> list[dict]:
    return [
        {"role": message.get("role"), "content": message.get("content")}
        for message in request.get("messages") or []
        if message.get("role") in ("user", "assistant") and message.get("content")
    ]


class _ToolCallIndexer:
    """Give each streamed tool call its own index. Some OpenAI-style providers reuse
    an index for parallel calls, which would merge their arguments into invalid JSON."""

    def __init__(self):
        self._by_slot: dict = {}
        self._next = 0

    def index_for(self, slot, call_id) -> int:
        known = self._by_slot.get(slot)
        if known is None or (call_id and known[0] and call_id != known[0]):
            known = (call_id, self._next)
            self._next += 1
            self._by_slot[slot] = known
        return known[1]


def _stream_chat_completions(
    client, config: ProviderConfig, request: dict, on_text, cancelled, relaxed: bool,
) -> ProviderReply:
    spec = config.spec
    options = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _system_text(spec, request)},
            *_chat_messages(request),
        ],
        "stream": True,
    }
    if spec.temperature is not None and not relaxed:
        options["temperature"] = spec.temperature
    tools = request.get("tools") or []
    if tools:
        forced = bool(request.get("must_call_tool")) and spec.forces_tool_calls
        options["tools"] = tools
        options["tool_choice"] = "required" if forced else "auto"
        if forced and spec.forced_tool_body:
            options["extra_body"] = spec.forced_tool_body

    reply = ProviderReply()
    fragments = []
    indexer = _ToolCallIndexer()
    with client.chat.completions.create(**options) as stream:
        for chunk in stream:
            if cancelled():
                raise ProviderCancelled
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                reply.text += delta.content
                on_text(delta.content)
            for call in getattr(delta, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                fragments.append({
                    "index": indexer.index_for(
                        getattr(call, "index", None), getattr(call, "id", None)
                    ),
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                })
            if choice.finish_reason == "length":
                reply.stop = "truncated"
            elif choice.finish_reason == "content_filter":
                reply.stop = "refused"
                reply.refusal_category = "content filter"
    reply.fragments = tuple(fragments)
    return reply


def _stream_openai_responses(
    client, config: ProviderConfig, request: dict, on_text, cancelled, relaxed: bool,
) -> ProviderReply:
    options = {
        "model": config.model,
        "instructions": _system_text(config.spec, request),
        "input": _chat_messages(request),
        "store": False,   # the default keeps every conversation on OpenAI's servers
        "stream": True,
    }
    tools = [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"].get("parameters") or {},
            "strict": False,
        }
        for tool in request.get("tools") or []
    ]
    if tools:
        options["tools"] = tools
        options["tool_choice"] = "auto"

    reply = ProviderReply()
    final = None
    with client.responses.create(**options) as stream:
        for event in stream:
            if cancelled():
                raise ProviderCancelled
            kind = getattr(event, "type", "")
            if kind == "response.output_text.delta":
                reply.text += event.delta
                on_text(event.delta)
            elif kind == "response.refusal.done":
                reply.stop = "refused"
            elif kind == "response.completed":
                final = event.response
            elif kind == "response.incomplete":
                final = event.response
                reply.stop = "truncated"
            elif kind in ("response.failed", "error"):
                raise ProviderError("failed", config)
    calls = [
        item for item in getattr(final, "output", None) or []
        if getattr(item, "type", "") == "function_call"
    ]
    reply.fragments = tuple(
        {"index": index, "name": item.name, "arguments": item.arguments}
        for index, item in enumerate(calls)
    )
    return reply


def _anthropic_messages(request: dict) -> list[dict]:
    messages = _chat_messages(request)
    # The history window can open on an assistant turn, which the API rejects.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def _stream_anthropic(
    client, config: ProviderConfig, request: dict, on_text, cancelled, relaxed: bool,
) -> ProviderReply:
    # No temperature, forced tool_choice or thinking setting: current models return a
    # 400 for each. eager_input_streaming stays off, because nothing shows tool input
    # as it streams and the API's own validation is worth keeping.
    options = {
        "model": config.model,
        "max_tokens": _ANTHROPIC_MAX_TOKENS,
        "system": _system_text(config.spec, request),
        "messages": _anthropic_messages(request),
    }
    tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": (
                tool["function"].get("parameters")
                or {"type": "object", "properties": {}}
            ),
        }
        for tool in request.get("tools") or []
    ]
    if tools:
        options["tools"] = tools
    if config.model.startswith(_ANTHROPIC_FALLBACK_MODELS) and not relaxed:
        manager = client.beta.messages.stream(
            betas=[_ANTHROPIC_FALLBACK_BETA], fallbacks="default", **options
        )
    else:
        manager = client.messages.stream(**options)

    reply = ProviderReply()
    with manager as stream:
        for event in stream:
            if cancelled():
                raise ProviderCancelled
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                reply.text += event.delta.text
                on_text(event.delta.text)
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        reply.stop = "refused"
        details = getattr(message, "stop_details", None)
        reply.refusal_category = str(getattr(details, "category", "") or "")
        return reply
    if message.stop_reason == "max_tokens":
        reply.stop = "truncated"
    blocks = list(message.content)
    # After a fallback, anything the declined model began is not part of the answer.
    last_fallback = max(
        (index for index, block in enumerate(blocks) if block.type == "fallback"),
        default=-1,
    )
    reply.fragments = tuple(
        {"index": index, "name": block.name, "arguments": json.dumps(block.input)}
        for index, block in enumerate(blocks[last_fallback + 1:])
        if block.type == "tool_use"
    )
    return reply


_ADAPTERS = {
    "openai_responses": _stream_openai_responses,
    "anthropic": _stream_anthropic,
    "chat_completions": _stream_chat_completions,
}


def stream_reply(
    config: ProviderConfig,
    request: dict,
    on_text: Callable[[str], None],
    cancelled: Callable[[], bool] = lambda: False,
    client_factory=_make_client,
) -> ProviderReply:
    """Stream one reply. A 400 before any output is retried once without the optional
    parameters, which are the part most likely to differ between models."""
    client = client_factory(config)
    stream = _ADAPTERS[config.spec.adapter]
    emitted = []

    def track(text: str):
        emitted.append(text)
        on_text(text)

    for relaxed in (False, True):
        try:
            return stream(client, config, request, track, cancelled, relaxed)
        except (ProviderCancelled, ProviderError):
            raise
        except Exception as error:
            kind = _failure_kind(error)
            if kind == "bad_request" and not relaxed and not emitted:
                continue
            raise ProviderError(kind, config) from None
    raise ProviderError("failed", config)


def list_models(config: ProviderConfig, client_factory=_make_client) -> list[str]:
    """Return the provider's chat models, which also proves the key works."""
    client = client_factory(config)
    try:
        names = [str(model.id).removeprefix("models/") for model in client.models.list()]
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(_failure_kind(error), config) from None
    return sorted(
        name for name in names
        if not any(word in name.lower() for word in _NON_CHAT_MODEL_WORDS)
    )


def check_connection(config: ProviderConfig, client_factory=_make_client) -> None:
    """Send a tiny request with a tool, as tools are what the assistant depends on."""
    stream_reply(
        config,
        {
            "system": "Reply with OK.",
            "messages": [{"role": "user", "content": "OK?"}],
            "tools": [_PING_TOOL],
        },
        lambda _text: None,
        client_factory=client_factory,
    )


def _post_json(endpoint: str, route: str, body: dict, timeout, post) -> dict:
    response = post(f"{endpoint}{route}", json=body, timeout=timeout)
    if response.status_code == 404:
        raise ServiceTooOld
    response.raise_for_status()
    return response.json()


def _reply_with_retry(
    config, model_request, on_text, on_status, cancelled, client_factory,
) -> ProviderReply:
    """Retry a transient failure once, and only if nothing has been shown yet."""
    emitted = []

    def track(text: str):
        emitted.append(text)
        on_text(text)

    for attempt in range(2):
        try:
            return stream_reply(config, model_request, track, cancelled, client_factory)
        except ProviderError as failure:
            if attempt or emitted or not failure.retryable:
                raise
            on_status({
                "level": "working",
                "stage": "retrying",
                "component": "byok",
                "code": "automatic_retry",
                "message": f"{failure.message} Retrying automatically...",
            })
            time.sleep(_RETRY_DELAY_SECONDS)
            if cancelled():
                raise ProviderCancelled
    raise ProviderError("failed", config)


def run_own_model_turn(
    endpoint: str,
    body: dict,
    config: ProviderConfig,
    *,
    on_text: Callable[[str], None],
    on_status: Callable[[dict], None] = lambda _info: None,
    cancelled: Callable[[], bool] = lambda: False,
    timeout=(10, 60),
    post=None,
    client_factory=_make_client,
) -> dict:
    """Run one turn. Exactly one provider call is made, however /finish behaves."""
    if post is None:
        import requests
        post = requests.post
    label = config.spec.label
    on_status({
        "level": "working",
        "stage": "connecting",
        "component": "modal",
        "message": "Connecting to Modal...",
    })
    prepared = _post_json(endpoint, "/prepare", body, timeout, post)
    if prepared.get("kind") == "final":
        on_status({
            "level": "working",
            "stage": "local_guidance",
            "component": "modal",
            "message": "Using the current BEHAV3D state...",
        })
        on_text(str(prepared.get("text") or ""))
        return {"calls": prepared.get("calls") or []}

    if prepared.get("retrieval_degraded"):
        on_status({
            "level": "degraded",
            "stage": "retrieval",
            "component": "retrieval",
            "code": "retrieval_unavailable",
            "message": (
                "The BEHAV3D guidance search is unavailable; continuing with "
                "built-in guidance."
            ),
        })
    on_status({
        "level": "working",
        "stage": "provider",
        "component": "byok",
        "message": f"Waiting for {label}...",
    })
    reply = _reply_with_retry(
        config, prepared, on_text, on_status, cancelled, client_factory
    )
    if reply.stop == "refused":
        # Skip /finish: partial text must not be recovered into a proposed change.
        return {"calls": [], "refused": True, "refusal_category": reply.refusal_category}

    result = {"calls": [], "truncated": reply.stop == "truncated"}
    on_status({
        "level": "working",
        "stage": "finishing",
        "component": "modal",
        "message": "Checking the proposed changes...",
    })
    finish_body = {
        **body,
        "content": reply.text,
        "tool_fragments": list(reply.fragments),
    }
    for attempt in range(_FINISH_ATTEMPTS):
        try:
            finished = _post_json(endpoint, "/finish", finish_body, timeout, post)
            result["calls"] = finished.get("calls") or []
            result["missing_required_tool"] = bool(finished.get("missing_required_tool"))
            return result
        except Exception:
            if attempt + 1 < _FINISH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    # Never call the provider again, and never apply what could not be checked.
    plan_call = prepared.get("plan_call")
    result["calls"] = [plan_call] if plan_call else []
    result["unchecked"] = bool(reply.fragments)
    return result


def _keyring():
    """Return the keyring module if it has a backend that really encrypts."""
    try:
        import keyring
        backend = keyring.get_keyring()
    except Exception:
        return None
    backends = getattr(backend, "backends", None)
    if backends is not None and not backends:
        return None
    name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
    if any(word in name for word in ("fail", "null", "plaintext")):
        return None
    return keyring


_session_keys: dict[str, str] = {}


def keychain_available() -> bool:
    return _keyring() is not None


def load_api_key(provider_id: str) -> ApiKey:
    keyring = _keyring()
    if keyring is not None:
        try:
            stored = keyring.get_password(KEYRING_SERVICE, provider_id)
            if stored:
                return ApiKey(stored)
        except Exception:
            pass
    return ApiKey(_session_keys.get(provider_id, ""))


def save_api_key(provider_id: str, key: ApiKey) -> bool:
    """Store a key, returning False when it can only last for this session."""
    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.set_password(KEYRING_SERVICE, provider_id, key.reveal())
            return True
        except Exception:
            pass
    _session_keys[provider_id] = key.reveal()
    return False


def delete_api_key(provider_id: str) -> None:
    _session_keys.pop(provider_id, None)
    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.delete_password(KEYRING_SERVICE, provider_id)
        except Exception:
            pass
