"""Shared factory functions for mswe_agents peer agents.

Provides load_base_config (minisweagent's mini.yaml), build_model
(provider-aware), and build_environment (Local/Docker). Provider routing:
"openai" → litellm (OpenAI Responses); "openrouter" → litellm Chat Completions
(preserves tool-call threading for resumed multi-turn extract/apply). Every
non-OpenAI backbone (deepseek, minimax, qwen, …) is served via OpenRouter.

`_is_reasoning_model` → reasoning_effort="high"; `_rejects_temperature` →
drop temperature. The two predicates are orthogonal: only OpenAI gpt-5/o-series
are both True. DeepSeek/MiniMax/Claude reason but accept temperature.
"""

from importlib.resources import files
from typing import Any

import litellm
import yaml

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

# DeepSeek is served via OpenRouter through litellm Chat Completions, so the
# model id carries an "openrouter/" prefix. litellm has no built-in pricing for
# that id, and minisweagent's LitellmModel raises on cost lookup — register the
# DeepSeek V4 prices under the openrouter-prefixed names. (flash: in 0.14e-6 /
# out 0.28e-6 / cache 0.0028e-6 per token.)
litellm.register_model({
    "openrouter/deepseek/deepseek-v4-flash": {
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "input_cost_per_token": 0.14e-6,
        "output_cost_per_token": 0.28e-6,
        "cache_read_input_token_cost": 0.0028e-6,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": True,
    },
    "openrouter/deepseek/deepseek-v4-pro": {
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "input_cost_per_token": 0.435e-6,
        "output_cost_per_token": 0.87e-6,
        "cache_read_input_token_cost": 0.003625e-6,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": True,
    },
    "openrouter/minimax/minimax-m3": {
        "max_tokens": 128_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 0.30e-6,
        "output_cost_per_token": 1.20e-6,
        "cache_read_input_token_cost": 0.06e-6,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": True,
    },
})

_REASONING_PREFIXES = (
    "o1-", "o1_", "o3-", "o3_", "o4-", "o4_",
    "gpt-5", "openai/gpt-5",
    "anthropic/", "claude-",
    "minimax/", "deepseek/",
)

# Only the OpenAI reasoning family rejects temperature (400). Others accept it.
_TEMPERATURE_INCOMPATIBLE_PREFIXES = (
    "o1-", "o1_", "o3-", "o3_", "o4-", "o4_",
    "gpt-5", "openai/gpt-5",
)

def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(name.startswith(p) for p in _REASONING_PREFIXES) or "reasoning" in name


def _rejects_temperature(model_name: str) -> bool:
    name = model_name.lower()
    return any(name.startswith(p) for p in _TEMPERATURE_INCOMPATIBLE_PREFIXES)


def load_base_config() -> dict:
    """Load minisweagent's bundled mini.yaml as the base agent config."""
    config_path = files("minisweagent.config") / "mini.yaml"
    return yaml.safe_load(config_path.read_text())


def _merge_model_section(cfg: dict, base_config: dict | None) -> dict:
    """Fill cfg from base_config['model'] for keys not already set."""
    if not base_config:
        return cfg
    base_model = base_config.get("model", {}) or {}
    for k, v in base_model.items():
        if k == "model_kwargs":
            cfg.setdefault("model_kwargs", {})
            for mk, mv in (v or {}).items():
                cfg["model_kwargs"].setdefault(mk, mv)
        else:
            cfg.setdefault(k, v)
    return cfg


def _build_litellm_config(
    model_name: str, temperature: float, max_tokens: int, api_key_env: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """litellm config for the openai path.

    ``reasoning_effort``, when set, is used verbatim; when None it defaults to
    "high" for reasoning models and unset otherwise.
    """
    import os

    cfg: dict[str, Any] = {
        "model_name": model_name,
        "model_class": "litellm",
        "model_kwargs": {
            "api_key": os.environ.get(api_key_env, ""),
        },
    }
    if reasoning_effort:
        cfg["model_kwargs"]["reasoning_effort"] = reasoning_effort
    elif _is_reasoning_model(model_name):
        cfg["model_kwargs"]["reasoning_effort"] = "high"
    if not _rejects_temperature(model_name):
        cfg["model_kwargs"]["temperature"] = temperature
    if max_tokens > 0:
        cfg["model_kwargs"]["max_tokens"] = max_tokens
    return cfg


def _build_openrouter_config(
    model_name: str, temperature: float, max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """OpenRouter Responses API config.

    ``model_kwargs`` is spread verbatim into the payload, so the top-level
    ``provider`` block routes the request: exclude quantized variants and forbid
    fallbacks by default. DeepSeek is pinned to its official provider (lab
    policy: avoid third-party quantized serving) with the quantization filter
    dropped, since that provider doesn't tag quantization. Mirrors
    ``utils.llm.llm_generation``.
    """
    import os

    provider_routing: dict[str, Any] = {
        "quantizations": ["unknown", "fp32", "fp16", "bf16", "fp8"],
        "allow_fallbacks": False,
    }
    if "deepseek" in model_name.lower():
        # NOTE: no `require_parameters` — with tools + temperature on the
        # Responses API it filters out every deepseek endpoint (HTTP 404
        # "No endpoints found that can handle the requested parameters").
        # `only` + `allow_fallbacks: False` already pin the official provider.
        provider_routing = {
            "only": ["deepseek"],
            "allow_fallbacks": False,
        }

    cfg: dict[str, Any] = {
        "model_name": model_name,
        "model_class": "openrouter_response",
        "model_kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "reasoning": {"effort": reasoning_effort or "high"},
            "provider": provider_routing,
        },
    }
    if not _rejects_temperature(model_name):
        cfg["model_kwargs"]["temperature"] = temperature
    if max_tokens > 0:
        cfg["model_kwargs"]["max_tokens"] = max_tokens
    return cfg


def _build_openrouter_litellm_config(
    model_name: str, temperature: float, max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """DeepSeek via OpenRouter through litellm Chat Completions (NOT the
    Responses API).

    The Responses API (``openrouter_response``) rejects resumed multi-turn
    conversations where an assistant ``tool_calls`` message is not immediately
    followed by its tool outputs (HTTP 400 "insufficient tool messages
    following tool_calls"), which breaks the extract-map / apply turns. The
    litellm Chat Completions path preserves the tool-call message threading the
    agent harness relies on — the same path the prior direct-DeepSeek runs
    used. We pin the official deepseek provider via ``extra_body`` and reach
    OpenRouter by prefixing the model id with ``openrouter/``.
    """
    import os

    or_model = model_name if model_name.startswith("openrouter/") else f"openrouter/{model_name}"
    model_kwargs: dict[str, Any] = {
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    }
    # DeepSeek is pinned to its official provider (lab policy: avoid third-party
    # quantized serving); other backbones use OpenRouter's default routing.
    if "deepseek" in model_name.lower():
        model_kwargs["extra_body"] = {
            "provider": {"only": ["deepseek"], "allow_fallbacks": False},
        }
    cfg: dict[str, Any] = {
        "model_name": or_model,
        "model_class": "litellm",
        "model_kwargs": model_kwargs,
    }
    if reasoning_effort:
        cfg["model_kwargs"]["reasoning_effort"] = reasoning_effort
    elif _is_reasoning_model(model_name):
        cfg["model_kwargs"]["reasoning_effort"] = "high"
    if not _rejects_temperature(model_name):
        cfg["model_kwargs"]["temperature"] = temperature
    if max_tokens > 0:
        cfg["model_kwargs"]["max_tokens"] = max_tokens
    return cfg


def build_model(
    *,
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = -1,
    base_config: dict | None = None,
    reasoning_effort: str | None = None,
):
    """Build a minisweagent Model from provider + name.

    ``reasoning_effort`` propagates to the SDK config; None keeps the default
    (high for reasoning models, unset otherwise).
    """
    if provider == "openai":
        cfg = _build_litellm_config(model_name, temperature, max_tokens, "OPENAILIKE_API_KEY",
                                    reasoning_effort=reasoning_effort)
    elif provider == "openrouter":
        # Chat Completions (litellm), not the Responses API — preserves tool-call
        # threading for resumed multi-turn (extract-map / apply). The Responses
        # API rejects resumed conversations where an assistant `tool_calls`
        # message isn't immediately followed by its outputs, which breaks those
        # turns. Applies to every OpenRouter backbone (deepseek, minimax, qwen, …).
        cfg = _build_openrouter_litellm_config(model_name, temperature, max_tokens,
                                               reasoning_effort=reasoning_effort)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r} (expected 'openai' or 'openrouter'). "
            "DeepSeek is served via 'openrouter' (lab policy: no direct DeepSeek API)."
        )

    cfg = _merge_model_section(cfg, base_config)
    # Use the config's (possibly rewritten) model_name as input — minisweagent's
    # get_model_name lets input_model_name override config, which would strip the
    # "openrouter/" prefix the deepseek litellm path adds and mis-route to the
    # direct DeepSeek API. cfg["model_name"] preserves the intended routing.
    return get_model(input_model_name=cfg["model_name"], config=cfg)


def build_environment(
    cwd: str,
    base_config: dict | None = None,
    timeout: int = 60,
    *,
    docker_image: str | None = None,
    host_dir: str | None = None,
    has_library: bool = False,
    forward_env: list[str] | None = None,
    mount_spec: list[tuple[str, str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    extra_run_args: list[str] | None = None,
):
    """Build a runtime environment: DockerEnvironment if ``docker_image`` is
    set, else a LocalEnvironment rooted at ``cwd``.

    Docker mounts: ``mount_spec`` (tuples of host_path, container_path, mode),
    when given, OVERRIDES the default paperbench layout:
        <host_dir>/paper/      → /home/paper       (:ro)
        <host_dir>/lib/        → /home/lib         (:ro, if has_library)
        <host_dir>/submission/ → /home/submission  (rw)
        <host_dir>/agent.env   → /home/agent.env   (rw)
    ``extra_env`` merges on top of base_config env; ``extra_run_args`` is
    appended after the mount block (``--platform``, ``--cpus``, etc.).
    """
    import os

    base_env: dict[str, str] = {}
    if base_config:
        base_env = (base_config.get("environment", {}) or {}).get("env", {}) or {}

    if docker_image:
        from minisweagent.environments.docker import DockerEnvironment

        env_dict = {
            **base_env,
            # Activate conda env "agent" (Dockerfile.base). Other images override
            # PATH/PYTHONPATH via extra_env.
            "PATH": "/opt/conda/envs/agent/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            # /home is the parent of /home/lib so `import lib` resolves from any cwd.
            "PYTHONPATH": "/home",
        }
        if extra_env:
            env_dict.update(extra_env)

        run_args: list[str] = ["--rm"]

        if mount_spec is not None:
            for host_path, container_path, mode in mount_spec:
                spec = f"{host_path}:{container_path}"
                if mode:
                    spec += f":{mode}"
                run_args += ["-v", spec]
        else:
            if not host_dir or not os.path.isdir(host_dir):
                raise ValueError(
                    f"docker_image set but host_dir does not exist: {host_dir!r}"
                )
            host_abs = os.path.abspath(host_dir)
            run_args += [
                "-v", f"{host_abs}/paper:/home/paper:ro",
                "-v", f"{host_abs}/submission:/home/submission",
                "-v", f"{host_abs}/agent.env:/home/agent.env",
            ]
            if has_library:
                run_args += ["-v", f"{host_abs}/lib:/home/lib:ro"]

        if extra_run_args:
            run_args += list(extra_run_args)

        return DockerEnvironment(
            image=docker_image,
            cwd=cwd,                       # container path
            env=env_dict,
            forward_env=forward_env or [
                "OPENAILIKE_API_KEY",
                "OPENROUTER_API_KEY",
                "HF_TOKEN",
            ],
            timeout=timeout,
            run_args=run_args,
            container_timeout="24h",
        )

    return LocalEnvironment(cwd=cwd, env=base_env, timeout=timeout)
