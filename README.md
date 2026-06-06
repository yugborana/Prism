# Prism — AI-Powered Multi-Agent Code Review System

Prism is a production-grade AI code review platform that analyzes GitHub pull requests using a fleet of specialized agents. It provides actionable feedback on security vulnerabilities, code quality issues, performance bottlenecks, and observability instrumentation gaps — posted directly as inline PR comments and suggestions.

## Features

### Multi-Agent Code Review
- **Security Agent:** OWASP Top 10 vulnerability detection, injection flaws, auth bypass, data exposure
- **Code Quality Agent:** Bug detection, code smells, design pattern violations, maintainability scoring
- **Performance Agent:** N+1 queries, algorithm complexity, memory leaks, missing connection pooling
- **Observability Agent:** Missing OpenTelemetry spans, logging gaps, metrics, event tracking

### 4-Step Reasoning Chain
Each agent uses a deliberation pipeline (Analyze → Generate → Critique → Refine) that produces significantly higher-quality reviews than single-shot LLM prompts by self-checking for false positives and inaccurate line numbers.

### GitHub Integration
- **Webhook-driven:** Automatic review on PR open/synchronize/reopen
- **Inline suggestions:** Code changes posted as GitHub suggestion blocks (one-click apply)
- **Comment triggers:** Re-run reviews or trigger specific analysis via PR comments (`prism check`, `prism dashboard`, `prism alerts`)
- **Dashboard/Alert suggestions:** With action markers for one-click creation in Grafana, Prometheus, Datadog, and Amplitude

### Production Infrastructure
- **Celery workers** for async background processing
- **Redis** task queue broker
- **PostgreSQL** audit trail for all reviews, decisions, and findings
- **Qdrant** vector database for code embedding search and context retrieval
- **Prometheus** metrics and structured logging
- **Helm charts** with HPA, NetworkPolicy, and ServiceAccount
- **Terraform** for AWS EKS cluster provisioning
- **Docker** image scanning (Trivy) and automated rollback on failed deploys

## Architecture

```
GitHub Webhook → FastAPI API → Celery Task Queue → ReviewOrchestrator
                                                        │
                          ┌─────────────────────────────┤
                          │                             │
                    ContextFetcher              Vector DB (Qdrant)
                          │
          ┌───────────────┼───────────────┐───────────────┐
          │               │               │               │
    SecurityAgent   QualityAgent   PerfAgent   ObservabilityAgent
          │               │               │               │
          └───────────────┼───────────────┘───────────────┘
                          │
                      Aggregator
                          │
                    GitHub PR Review
```

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Redis 7+
- PostgreSQL 15+ (optional, for audit trail)
- Qdrant (optional, for vector search context)
- GitHub App or Personal Access Token
- LLM API key (Groq, Anthropic, OpenAI, Gemini, or local Ollama)

## Quick Start (Local Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yugborana/Prism.git
   cd Prism
   ```

2. **Create environment file:**
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your API keys
   ```

3. **Start services with Docker Compose:**
   ```bash
   cd backend
   docker-compose up -d
   ```

4. **Run the development server:**
   ```bash
   pip install -e ".[dev]"
   python -m uvicorn main:app --reload --port 8000
   ```

5. **Test a review manually:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/webhooks/test-review \
     -H "Content-Type: application/json" \
     -d '{"repo": "owner/repo", "pr_number": 123}'
   ```

## Configuration

All configuration is via environment variables (loaded from `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub PAT or App installation token | (required) |
| `GITHUB_WEBHOOK_SECRET` | HMAC secret for webhook verification | (required) |
| `LLM_PROVIDER` | LLM provider: `groq`, `anthropic`, `openai`, `gemini`, `ollama` | `groq` |
| `GROQ_API_KEY` | Groq API key | |
| `ANTHROPIC_API_KEY` | Anthropic API key | |
| `OPENAI_API_KEY` | OpenAI API key | |
| `GEMINI_API_KEY` | Google Gemini API key | |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `DATABASE_URL` | PostgreSQL URL for audit trail | `postgresql+asyncpg://prism:prism@localhost:5432/prism` |
| `QDRANT_URL` | Qdrant vector DB URL | `http://localhost:6333` |
| `MAX_DIFF_SIZE` | Max cumulative diff size (per-file cap) | `50000` |
| `PRD_FILE_PATH` | Path to Product Requirements Document | |
| `ENVIRONMENT` | `development`, `staging`, `production` | `development` |

## CI/CD Integration

Prism includes GitHub Actions workflows for:

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push to main, PRs | Lint, test, build, scan, deploy |
| `prism-pr-trigger.yml` | PR opened/updated | Run full AI code review |
| `prism-comment-trigger.yml` | PR comment | `prism check`, `prism dashboard`, `prism alerts` |
| `prism-dashboard-creation.yml` | PR comment | Create dashboards in Grafana/Datadog/Amplitude |
| `prism-alert-creation.yml` | PR comment | Create alerts in Prometheus/Datadog |

### Required GitHub Secrets

- `GROQ_API_KEY` (or your preferred LLM provider's key)
- `AWS_ROLE_ARN` (for ECR/EKS deployment)
- `GRAFANA_SERVICE_ACCOUNT_TOKEN` (for dashboard creation)
- `DATADOG_API_KEY` / `DATADOG_APP_KEY` (for Datadog integration)

## Project Structure

```
Prism/
├── backend/
│   ├── agents/              # Review agent fleet
│   │   ├── security_agent.py
│   │   ├── quality_agent.py
│   │   ├── performance_agent.py
│   │   ├── observability_agent.py
│   │   ├── aggregator.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── reasoning.py
│   │   └── base_agent.py
│   ├── api/                 # FastAPI routes
│   ├── services/            # GitHub, comment, vector services
│   ├── orchestrator/        # DAG-based review pipeline
│   ├── workers/             # Celery background tasks
│   ├── db/                  # PostgreSQL + Qdrant
│   ├── observability/       # Logging + Prometheus metrics
│   └── utils/               # Config, LLM factory
├── infra/
│   ├── helm/prism/          # Helm chart (HPA, NetworkPolicy, ServiceAccount)
│   └── terraform/           # AWS EKS provisioning
├── monitoring/
│   ├── grafana/             # Grafana dashboard configs
│   └── prometheus/          # Prometheus alert rules
└── .github/workflows/       # CI/CD + PR review automation
```

## Troubleshooting

### LLM Errors
- Verify your API key is set correctly in `.env`
- Check provider-specific rate limits (Groq has generous free tier)
- Use `--max-diff-size` to reduce context for large PRs

### Large PRs
- Files exceeding `MAX_DIFF_SIZE` are skipped individually (not the whole diff)
- Consider splitting large PRs into focused changes

### GitHub API Rate Limits
- Use a GitHub App installation token for higher rate limits
- The webhook handler uses `continue-on-error: true` for resilience

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.
