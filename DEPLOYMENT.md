# Prism Deployment Guide

This guide covers deploying Prism to production on AWS EKS.

## Table of Contents

- [Local Development](#local-development)
- [AWS EKS Deployment](#aws-eks-deployment)
- [GitHub App Configuration](#github-app-configuration)
- [Secret Management](#secret-management)
- [Monitoring Setup](#monitoring-setup)

## Local Development

### Prerequisites
- Docker Desktop
- Python 3.12+
- An LLM API key (Groq recommended for free tier)

### Steps

1. **Start infrastructure services:**
   ```bash
   cd backend
   docker-compose up -d redis qdrant postgres
   ```

2. **Create `.env` from example:**
   ```bash
   cp .env.example .env
   # Set GITHUB_TOKEN, LLM_PROVIDER, and API key
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the API server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Run the Celery worker (separate terminal):**
   ```bash
   celery -A workers.celery_app worker --loglevel=info
   ```

6. **Test a review:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/webhooks/test-review \
     -H "Content-Type: application/json" \
     -d '{"repo": "owner/repo", "pr_number": 123}'
   ```

### Exposing Locally (for GitHub Webhooks)

Use a tunnel to expose the local server:
```bash
# With Cloudflare Tunnel (recommended):
cloudflared tunnel --url http://localhost:8000

# Or with ngrok:
ngrok http 8000
```

Set the tunnel URL as the GitHub webhook endpoint.

## AWS EKS Deployment

### 1. Provision Infrastructure (Terraform)

```bash
cd infra/terraform
terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan
```

This creates:
- EKS cluster with managed node groups
- ECR repository for Docker images
- RDS PostgreSQL instance
- ElastiCache Redis cluster
- VPC with public/private subnets

### 2. Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t prism-backend backend/
docker tag prism-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/prism-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/prism-backend:latest
```

### 3. Deploy with Helm

```bash
# Update kubeconfig
aws eks update-kubeconfig --name prism-cluster --region us-east-1

# Deploy
helm upgrade --install prism infra/helm/prism \
  --set image.repository=<account>.dkr.ecr.us-east-1.amazonaws.com/prism-backend \
  --set image.tag=latest \
  --set secrets.geminiApiKey=$GEMINI_API_KEY \
  --set secrets.githubWebhookSecret=$GITHUB_WEBHOOK_SECRET \
  --set secrets.grafanaServiceAccountToken=$GRAFANA_TOKEN \
  --namespace prism \
  --create-namespace \
  --wait
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n prism

# Check health
kubectl port-forward svc/prism 8000:8000 -n prism
curl http://localhost:8000/api/health
```

## GitHub App Configuration

### Required Permissions

| Permission | Access | Reason |
|-----------|--------|--------|
| Pull requests | Read & Write | Post review comments and suggestions |
| Contents | Read | Fetch PR diffs and file contents |
| Issues | Read & Write | Post summary comments, read webhooks |
| Metadata | Read | Repository information |

### Webhook Events

Subscribe to these events:
- `pull_request` — Trigger automatic reviews
- `issue_comment` — Handle `prism check`, `prism dashboard`, `prism alerts` commands

### Webhook URL

Set to: `https://<your-domain>/api/v1/webhooks/github`

## Secret Management

### For CI/CD (GitHub Actions)

Add these as repository secrets:
- `AWS_ROLE_ARN` — OIDC role for ECR/EKS access
- `GROQ_API_KEY` (or preferred LLM provider key)
- `GRAFANA_SERVICE_ACCOUNT_TOKEN`
- `GRAFANA_URL`
- `DATADOG_API_KEY`
- `DATADOG_APP_KEY`

### For Production (Kubernetes)

Use AWS Secrets Manager + External Secrets Operator, or set via Helm:
```bash
helm upgrade prism infra/helm/prism \
  --set secrets.geminiApiKey=$(aws secretsmanager get-secret-value --secret-id prism/gemini --query SecretString --output text)
```

## Monitoring Setup

### Prometheus

Prism exposes metrics at `/api/metrics`:
- `prism_review_duration_seconds` — Review pipeline latency
- `prism_agent_task_duration_seconds` — Per-agent execution time
- `prism_findings_total` — Findings by agent and severity
- `prism_webhook_requests_total` — Webhook event counts

The Prometheus config is in `monitoring/prometheus/prometheus.yml`.

### Grafana

Import the dashboards from `monitoring/grafana/dashboards/` into your Grafana instance.

### Alerting

Prism can generate Prometheus alert rules and Datadog monitors based on PR analysis.
Use the `prism alerts` comment trigger to generate and create alert rules automatically.
