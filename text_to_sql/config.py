"""Central configuration for the TransitIQ Text-to-SQL generation and validation pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderType = Literal["gemini", "openai", "mock"]

# Estimated USD pricing per 1,000,000 tokens (input, output) for cost tracking (§11.2 & §17 PRD)
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mock-model": (0.0, 0.0),
}


class TextToSQLSettings(BaseSettings):
    """Settings for LLM generation, SQL validation guardrails, and query execution."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProviderType = Field(
        default="gemini",
        validation_alias="LLM_PROVIDER",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.0,
        validation_alias="LLM_TEMPERATURE",
    )
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # SQL validation & security guardrails (§6 F-004 & §16 PRD)
    max_row_limit: int = Field(default=1000, validation_alias="TRANSITIQ_MAX_ROW_LIMIT")
    max_joins: int = Field(default=6, validation_alias="TRANSITIQ_MAX_JOINS")
    query_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="TRANSITIQ_QUERY_TIMEOUT_SECONDS",
    )
    prompt_version: str = Field(
        default="v1_full_schema",
        validation_alias="TRANSITIQ_PROMPT_VERSION",
    )

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute estimated USD cost for a single LLM invocation."""
        in_rate, out_rate = MODEL_PRICING_PER_1M_TOKENS.get(
            self.llm_model,
            (0.15, 0.60),
        )
        cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
        return round(cost, 6)


t2s_settings = TextToSQLSettings()
