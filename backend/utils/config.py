"""
Prism Application Configuration.

Pattern: Pydantic Settings with .env file loading and validation.

All environment variables are documented in .env.example
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Central configuration for the entire Prism backend.
    Values are loaded from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── GitHub App Integration ────────────────────────────────────────────
    github_webhook_secret: str = Field(default="", description="HMAC secret for webhook signature verification")
    github_app_id: str = Field(default="", description="GitHub App ID")
    github_app_slug: str = Field(default="", description="GitHub App slug for installation URL")
    github_app_private_key_path: str = Field(default="", description="Path to GitHub App private key PEM file or the key content itself")
    github_token: str = Field(default="", description="GitHub Personal Access Token for API access")

    # ── LLM Provider Configuration ────────────────────────────────────────
    llm_provider: str = Field(default="groq", description="Default LLM provider: gemini | openai | anthropic | groq | bedrock | ollama")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    aws_region: str = Field(default="us-east-1", description="AWS region for Bedrock")

    # ── LLM Model Names (per provider) ────────────────────────────────────
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", description="Anthropic model name")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    bedrock_model: str = Field(default="anthropic.claude-sonnet-4-20250514-v1:0", description="Bedrock model ID")

    # ── Ollama (Local LLM) ────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://host.docker.internal:11434/v1", description="Ollama server base URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model name")

    # ── Embedding Configuration ───────────────────────────────────────────
    embedding_model: str = Field(default="all-minilm", description="Embedding model for Qdrant indexing (via Ollama)")
    embedding_dim: int = Field(default=384, description="Embedding vector dimension (all-minilm = 384)")

    # ── Vector Database (Qdrant) ──────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (for cloud)")

    # ── PostgreSQL (Permanent Audit Trail) ──────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://prism:prism@localhost:5432/prism",
        description="PostgreSQL connection URL for audit trail storage",
    )

    # ── Redis (Celery Task Queue Broker) ────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL for Celery")

    # ── LLM Limits ────────────────────────────────────────────────────────
    max_retries: int = Field(default=3, description="Max LLM call retries per agent")
    request_timeout: int = Field(default=120, description="LLM request timeout in seconds")
    max_diff_size: int = Field(default=50000, description="Max cumulative diff size in chars. Files exceeding this are skipped individually.")
    prd_file_path: str = Field(default="", description="Path to Product Requirements Document. Passed to LLM for product-aware review.")

    # ── Monitoring Integrations ─────────────────────
    grafana_url: str = Field(default="", description="Grafana server URL (e.g., https://grafana.example.com)")
    grafana_service_account_token: str = Field(default="", description="Grafana service account token for API access")
    datadog_api_key: str = Field(default="", description="Datadog API key")
    datadog_app_key: str = Field(default="", description="Datadog Application key")
    datadog_host: str = Field(default="api.datadoghq.com", description="Datadog API host (e.g., api.ap1.datadoghq.com for Asia Pacific)")
    amplitude_api_key: str = Field(default="", description="Amplitude API key")
    amplitude_secret_key: str = Field(default="", description="Amplitude secret key")
    prometheus_config_path: str = Field(default="monitoring/prometheus/rules", description="Path for Prometheus alert rule files")

    # ── Observability ─────────────────────────────────────────────────────
    environment: str = Field(default="development", description="development | staging | production")
    otel_endpoint: str = Field(default="http://localhost:4317", description="OpenTelemetry collector gRPC endpoint")
    prometheus_port_api: int = Field(default=9091, description="Prometheus metrics port for the API process")
    prometheus_port_worker: int = Field(default=9092, description="Prometheus metrics port for the Celery worker process")
    log_level: str = Field(default="info", description="Logging level: debug | info | warning | error")

    # ── Server ────────────────────────────────────────────────────────────
    port: int = Field(default=8000, description="FastAPI server port")
    temp_repo_dir: str = Field(default="./temp_repos", description="Directory for temporary repo clones")
    cors_origins: str = Field(default="http://localhost:3000", description="Comma-separated allowed CORS origins")

    # ── Cloudflare Tunnel ─────────────────────────────────────────────────
    tunnel_token: str = Field(default="", description="Cloudflare Tunnel token for production")

    # ── API Authentication ────────────────────────────────────────────────
    prism_api_key: str = Field(default="", description="API key for authenticating internal endpoints (/test-review, /metrics, /dashboards, /alerts). Leave empty to allow unauthenticated access (dev only).")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_model_for_provider(self, provider: str | None = None) -> str:
        """Return the configured model name for a given provider."""
        p = provider or self.llm_provider
        model_map = {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "groq": self.groq_model,
            "bedrock": self.bedrock_model,
            "ollama": self.ollama_model,
        }
        return model_map.get(p, self.gemini_model)

    def get_api_key_for_provider(self, provider: str | None = None) -> str:
        """Return the configured API key for a given provider."""
        p = provider or self.llm_provider
        key_map = {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "bedrock": "",  # Bedrock uses IAM, not API keys
            "ollama": "",   # Ollama is local, no API key needed
        }
        return key_map.get(p, "")

# Singleton — imported by the rest of the application
settings = Settings()
