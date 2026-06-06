# Prism — Product Requirements Document

> This document is passed to the LLM during PR reviews to provide product context.
> Customize it for your team's specific observability standards and coding guidelines.

## Product Overview

Prism is a multi-agent AI code review system that automatically analyzes GitHub pull
requests for security vulnerabilities, code quality issues, performance bottlenecks,
and observability instrumentation gaps.

## Review Standards

### Security
- All user input must be validated and sanitized
- No secrets or API keys in code (use environment variables or secret managers)
- Authentication and authorization must be verified for every endpoint
- SQL queries must use parameterized statements
- Error messages must not expose internal details to users

### Code Quality
- Functions should follow Single Responsibility Principle
- No duplicated code blocks (DRY)
- Error handling must include meaningful context
- All public functions should have docstrings
- Naming conventions should be clear and descriptive

### Performance
- Database queries should avoid N+1 patterns (use eager loading)
- Large collections should be paginated
- Expensive operations should be cached where appropriate
- I/O operations should be async where the framework supports it
- No unbounded queries (always use LIMIT)

### Observability
- **Every API endpoint** must have OpenTelemetry spans with relevant attributes
- **Every error path** must log with structured context (not just the error message)
- **Every background task** must emit start/end metrics
- **User-facing actions** should have analytics event tracking
- **Performance-critical paths** should have histogram metrics for latency

## Technology Stack

- **Backend:** Python 3.12, FastAPI, Celery
- **Database:** PostgreSQL (audit trail), Qdrant (vector search)
- **Cache/Queue:** Redis
- **Infrastructure:** AWS EKS, Helm, Terraform
- **Monitoring:** Prometheus, Grafana
- **Analytics:** Amplitude (event tracking)

## Definition of Done (for PRs)

A PR is ready to merge when:
1. All review agents report zero CRITICAL/HIGH findings
2. Observability instrumentation score ≥ 7/10
3. Unit tests cover new code paths
4. No new security vulnerabilities introduced
5. Performance impact assessed for data-heavy operations
