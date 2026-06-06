# Prism Terraform — Full AWS Production Infrastructure
# Usage: terraform init && terraform plan && terraform apply
#
# Creates:
#   VPC           — Private network with public + private subnets
#   EKS           — Kubernetes cluster for API + Worker containers
#   ECR           — Private Docker image registry
#   RDS           — Managed PostgreSQL for audit trails
#   ElastiCache   — Managed Redis for Celery task queue
#   Secrets Mgr   — Secure vault for API keys and tokens
#   IAM           — Least-privilege roles for EKS pods (IRSA)
#   S3            — Terraform state storage (backend)

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "prism-terraform-state"
    key    = "eks/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ────────────────────────────────────────────────────────────
variable "aws_region" {
  default = "us-east-1"
}

variable "cluster_name" {
  default = "prism-cluster"
}

variable "db_password" {
  description = "PostgreSQL master password — pass via TF_VAR_db_password or -var"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub PAT for PR API access"
  type        = string
  sensitive   = true
}

variable "github_webhook_secret" {
  description = "HMAC secret for webhook signature verification"
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq LLM API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "prism_api_key" {
  description = "API key for authenticating internal Prism endpoints"
  type        = string
  sensitive   = true
}

# ── VPC ──────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "prism-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true  # Cost optimization for non-prod

  tags = {
    Project = "prism"
  }
}

# ── EKS Cluster ──────────────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  eks_managed_node_groups = {
    prism_nodes = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 5
      desired_size   = 2
    }
  }

  tags = {
    Project = "prism"
  }
}

# ── ECR Repository ───────────────────────────────────────────────────────
resource "aws_ecr_repository" "prism" {
  name                 = "prism-backend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = "prism"
  }
}

# ── RDS PostgreSQL (Audit Trail) ─────────────────────────────────────────
resource "aws_db_subnet_group" "prism" {
  name       = "prism-db-subnet"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Project = "prism"
  }
}

resource "aws_security_group" "rds" {
  name   = "prism-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    description = "PostgreSQL from private subnets"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = "prism"
  }
}

resource "aws_db_instance" "prism_postgres" {
  identifier     = "prism-postgres"
  engine         = "postgres"
  engine_version = "15.7"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = "prism"
  username = "prism"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.prism.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Backups
  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # Don't expose publicly
  publicly_accessible = false

  # Prevent accidental deletion
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "prism-postgres-final"

  tags = {
    Project = "prism"
  }
}

# ── ElastiCache Redis (Celery Broker) ────────────────────────────────────
resource "aws_elasticache_subnet_group" "prism" {
  name       = "prism-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name   = "prism-redis-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    description = "Redis from private subnets"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = "prism"
  }
}

resource "aws_elasticache_cluster" "prism_redis" {
  cluster_id           = "prism-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.prism.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Project = "prism"
  }
}

# ── AWS Secrets Manager ──────────────────────────────────────────────────
# Stores all sensitive config (API keys, tokens) securely.
# Your EKS pods read these at startup via the IRSA role below.

resource "aws_secretsmanager_secret" "prism_secrets" {
  name        = "prism/production/secrets"
  description = "Prism production secrets (GitHub token, LLM keys, etc.)"

  tags = {
    Project = "prism"
  }
}

resource "aws_secretsmanager_secret_version" "prism_secrets_value" {
  secret_id = aws_secretsmanager_secret.prism_secrets.id
  secret_string = jsonencode({
    GITHUB_TOKEN          = var.github_token
    GITHUB_WEBHOOK_SECRET = var.github_webhook_secret
    GROQ_API_KEY          = var.groq_api_key
    PRISM_API_KEY         = var.prism_api_key
    DATABASE_URL          = "postgresql+asyncpg://prism:${var.db_password}@${aws_db_instance.prism_postgres.endpoint}/prism"
    REDIS_URL             = "redis://${aws_elasticache_cluster.prism_redis.cache_nodes[0].address}:6379/0"
  })
}

# ── IAM Role for EKS Pods (IRSA) ─────────────────────────────────────────
# This role is attached to the Kubernetes ServiceAccount used by Prism pods.
# It gives them permission to read secrets from Secrets Manager — and nothing else.

data "aws_iam_policy_document" "prism_pod_policy" {
  # Allow reading Prism secrets
  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.prism_secrets.arn]
  }

  # Allow pulling images from ECR
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "prism_pod_policy" {
  name   = "prism-pod-policy"
  policy = data.aws_iam_policy_document.prism_pod_policy.json

  tags = {
    Project = "prism"
  }
}

module "prism_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "prism-pod-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:prism-sa"]
    }
  }

  role_policy_arns = {
    prism = aws_iam_policy.prism_pod_policy.arn
  }

  tags = {
    Project = "prism"
  }
}

# ── Outputs ──────────────────────────────────────────────────────────────
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  value = aws_ecr_repository.prism.repository_url
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.prism_redis.cache_nodes[0].address
}

output "rds_endpoint" {
  description = "PostgreSQL endpoint for DATABASE_URL"
  value       = aws_db_instance.prism_postgres.endpoint
}

output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.prism_secrets.arn
}

output "pod_role_arn" {
  description = "IAM Role ARN to set in Helm values.serviceAccount.roleArn"
  value       = module.prism_irsa_role.iam_role_arn
}
