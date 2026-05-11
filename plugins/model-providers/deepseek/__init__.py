"""DeepSeek provider profile.

DeepSeek API supports thinking mode which requires ``extra_body`` with
``{"thinking": {"type": "enabled"}}`` and top-level ``reasoning_effort``.

Refs: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class DeepseekProfile(ProviderProfile):
    """DeepSeek — extra_body.thinking + top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body = {}
        top_level = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config → thinking enabled, default effort
            extra_body["thinking"] = {"type": "enabled"}
            top_level["reasoning_effort"] = "high"
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        # Enabled
        extra_body["thinking"] = {"type": "enabled"}
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort in ("low", "medium", "high", "max"):
            top_level["reasoning_effort"] = effort
        else:
            top_level["reasoning_effort"] = "high"

        return extra_body, top_level


deepseek = DeepseekProfile(
    name="deepseek",
    aliases=("deepseek-chat",),
    env_vars=("DEEPSEEK_API_KEY",),
    display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API",
    signup_url="https://platform.deepseek.com/",
    fallback_models=(
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    base_url="https://api.deepseek.com/v1",
)

register_provider(deepseek)
