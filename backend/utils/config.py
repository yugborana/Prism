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
    github_app_private_key_path: str = Field(default="", description="Path to GitHub App private key PEM file (local dev)")
    github_app_private_key: str = Field(default="", description="GitHub App private key PEM content (production — injected from AWS Secrets Manager)")
    github_token: str = Field(default="", description="GitHub PAT fallback for local testing only. Not used in production.")

    # ── AWS ───────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region for Bedrock and Secrets Manager")

    # ── LLM Provider Configuration ────────────────────────────────────────
    llm_provider: str = Field(default="groq", description="Default LLM provider: gemini | openai | anthropic | groq | bedrock | ollama")
    # ── LLM Gateway (LiteLLM Proxy) ───────────────────────────────────────


    # ── Ollama (Local LLM) ────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://host.docker.internal:11434/v1", description="Ollama server base URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model name")

    # ── Embedding Configuration ───────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model for Qdrant indexing (sentence-transformers)")
    embedding_dim: int = Field(default=384, description="Embedding vector dimension (all-MiniLM-L6-v2 = 384)")

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
    grafana_url: str = Field(default="", description="Grafana server URL (user-provided, e.g., https://grafana.example.com)")
    grafana_service_account_token: str = Field(default="", description="Grafana service account token for API access")
    datadog_api_key: str = Field(default="", description="Datadog API key")
    datadog_app_key: str = Field(default="", description="Datadog Application key")
    datadog_host: str = Field(default="api.datadoghq.com", description="Datadog API host (e.g., api.ap1.datadoghq.com for Asia Pacific)")
    amplitude_api_key: str = Field(default="", description="Amplitude API key")
    amplitude_secret_key: str = Field(default="", description="Amplitude secret key")

    # ── Observability ─────────────────────────────────────────────────────
    environment: str = Field(default="development", description="development | staging | production")
    log_level: str = Field(default="info", description="Logging level: debug | info | warning | error")

    # ── Server ────────────────────────────────────────────────────────────
    port: int = Field(default=8000, description="FastAPI server port")
    temp_repo_dir: str = Field(default="./temp_repos", description="Directory for temporary repo clones")
    cors_origins: str = Field(default="http://localhost:3000", description="Comma-separated allowed CORS origins")

    # ── Connection Pools ──────────────────────────────────────────────────
    redis_pool_min: int = Field(default=2, description="Minimum Redis connection pool size")
    redis_pool_max: int = Field(default=10, description="Maximum Redis connection pool size")
    httpx_max_connections: int = Field(default=20, description="Max total connections in shared httpx client")
    httpx_max_keepalive: int = Field(default=5, description="Max keep-alive connections in shared httpx client")
    asyncpg_pool_size: int = Field(default=2, description="asyncpg base pool size (constrained for t2.micro)")
    asyncpg_max_overflow: int = Field(default=5, description="asyncpg max overflow connections")
    uvicorn_workers: int = Field(default=2, description="Number of Uvicorn workers (limited for t2.micro 1GB RAM)")

    # ── Repo Indexing (Cursor-style codebase indexing) ────────────────────
    repo_index_max_files: int = Field(default=5000, description="Skip indexing repos with more files than this")
    repo_clone_max_age_hours: int = Field(default=24, description="Auto-cleanup repo clones older than this (hours)")
    chunk_embedding_cache_ttl: int = Field(default=604800, description="Embedding cache TTL in seconds (7 days)")
    simhash_similarity_threshold: int = Field(default=25, description="SimHash Hamming distance threshold for index reuse")
    repo_index_ttl_days: int = Field(default=30, description="Delete repo indexes not updated in this many days")

    # ── Cloudflare Tunnel ─────────────────────────────────────────────────
    tunnel_token: str = Field(default="", description="Cloudflare Tunnel token for production")

    # ── API Authentication ────────────────────────────────────────────────
    prism_api_key: str = Field(default="", description="API key for authenticating internal endpoints (/test-review, /metrics, /dashboards, /alerts). Leave empty to allow unauthenticated access (dev only).")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @classmethod
    def load_production_secrets(cls):
        """Load secrets from SSM on EC2 (replaces Secrets Manager + ESO)."""
        import os
        if os.getenv("ENVIRONMENT") == "production":
            import boto3
            client = boto3.client("ssm", region_name=os.getenv("AWS_REGION", "us-east-1"))
            try:
                def get_param(name):
                    return client.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
                os.environ["GITHUB_WEBHOOK_SECRET"] = get_param("/prism/webhook_secret")
                os.environ["GITHUB_APP_PRIVATE_KEY"] = get_param("/prism/github_app_private_key")
                os.environ["GROQ_API_KEY"] = get_param("/prism/groq_api_key")
                os.environ["PRISM_API_KEY"] = get_param("/prism/prism_api_key")
                os.environ["DATABASE_URL"] = get_param("/prism/db_url")
            except Exception as e:
                print(f"Warning: Failed to load secrets from SSM: {e}")


# Load production secrets before instantiating Settings
Settings.load_production_secrets()

# Singleton — imported by the rest of the application
settings = Settings()
