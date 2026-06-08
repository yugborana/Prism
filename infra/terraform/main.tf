# Prism AWS Free Tier Infrastructure
# Creates EC2 (t2.micro), RDS (db.t3.micro), Security Groups, IAM, and SSM.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "prism-tfstate-yugborana-88"
    key    = "free-tier/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

# ── Variables ────────────────────────────────────────────────────────────
variable "db_password" {
  description = "PostgreSQL master password"
  type        = string
  sensitive   = true
}

variable "github_app_id" { type = string }
variable "github_app_private_key" { type = string, sensitive = true }
variable "github_webhook_secret" { type = string, sensitive = true }
variable "groq_api_key" { type = string, sensitive = true }
variable "prism_api_key" { type = string, sensitive = true }

variable "ssh_allowed_cidrs" {
  description = "CIDR blocks allowed to SSH into EC2 (restrict to your IP or GitHub Actions runner IPs)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # OVERRIDE this in terraform.tfvars for production!
}

# ── Data Sources ─────────────────────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── Security Groups ──────────────────────────────────────────────────────
resource "aws_security_group" "ec2_sg" {
  name        = "prism-ec2-sg"
  description = "Allow inbound traffic to EC2"
  vpc_id      = data.aws_vpc.default.id

  # Allow SSH — restrict in terraform.tfvars to your IP / GitHub Actions runner IPs
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidrs
    description = "SSH access — override ssh_allowed_cidrs in tfvars"
  }

  # Allow HTTP (API)
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "prism-rds-sg"
  description = "Allow EC2 to access RDS"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── IAM Role for EC2 ─────────────────────────────────────────────────────
resource "aws_iam_role" "prism_ec2_role" {
  name = "prism-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "prism_ec2_policy" {
  name = "prism-ec2-policy"
  role = aws_iam_role.prism_ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = "arn:aws:ssm:us-east-1:*:parameter/prism/*"
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:us-east-1:*:log-group:/prism/*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "prism_ec2_profile" {
  name = "prism-ec2-profile"
  role = aws_iam_role.prism_ec2_role.name
}

# ── EC2 Instance ─────────────────────────────────────────────────────────
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

resource "aws_instance" "prism_app" {
  ami           = data.aws_ami.amazon_linux_2023.id
  instance_type = "t2.micro" # Free Tier eligible
  
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.prism_ec2_profile.name

  # Note: You need an existing key pair named 'prism-key' in AWS
  key_name = "prism-key"

  # EBS root volume — stay within 30GB free tier
  root_block_device {
    volume_size           = 30     # 30GB max free tier EBS
    volume_type           = "gp2"  # gp2 is explicitly free tier eligible
    delete_on_termination = true   # Prevent orphaned EBS charges on destroy
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    dnf update -y
    dnf install -y docker git
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ec2-user
    
    # Install Docker Compose v2 plugin (modern path)
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64 \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    
    # Create 2GB swap — critical for t2.micro (1GB RAM) running 4+ containers
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
  EOF

  tags = { Name = "prism-app" }
}

# ── Elastic IP ─────────────────────────────────────────────────────────
# One EIP is free when attached to a running instance.
# Gives a stable IP so GitHub webhook URL and CI SSH target never change.
resource "aws_eip" "prism_eip" {
  instance = aws_instance.prism_app.id
  domain   = "vpc"
  tags     = { Name = "prism-eip" }
}

# ── RDS Free Tier ────────────────────────────────────────────────────────
resource "aws_db_instance" "prism_postgres" {
  identifier     = "prism-postgres"
  engine         = "postgres"
  engine_version = "15.7"
  instance_class = "db.t3.micro" # Free Tier eligible

  allocated_storage     = 20     # Max free tier
  max_allocated_storage = 0      # Disable autoscaling for cost control
  storage_type          = "gp2"  # gp2 is explicitly covered by free tier (gp3 is NOT)
  storage_encrypted     = false  # Free-tier KMS has limits; acceptable for MVP
  multi_az              = false  # Multi-AZ is NOT free tier eligible

  db_name  = "prism"
  username = "prism"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  publicly_accessible    = false
  skip_final_snapshot    = true  # Easy teardown
  
  # Prevent accidental deletion in production
  deletion_protection = false    # Set to true once stable
}

# ── SSM Parameters (Secrets) ─────────────────────────────────────────────
resource "aws_ssm_parameter" "github_app_id" {
  name  = "/prism/github_app_id"
  type  = "String"
  value = var.github_app_id
}

resource "aws_ssm_parameter" "github_app_private_key" {
  name  = "/prism/github_app_private_key"
  type  = "SecureString"
  value = var.github_app_private_key
}

resource "aws_ssm_parameter" "webhook_secret" {
  name  = "/prism/webhook_secret"
  type  = "SecureString"
  value = var.github_webhook_secret
}

resource "aws_ssm_parameter" "groq_api_key" {
  name  = "/prism/groq_api_key"
  type  = "SecureString"
  value = var.groq_api_key
}

resource "aws_ssm_parameter" "prism_api_key" {
  name  = "/prism/prism_api_key"
  type  = "SecureString"
  value = var.prism_api_key
}

resource "aws_ssm_parameter" "db_url" {
  name  = "/prism/db_url"
  type  = "SecureString"
  # asyncpg dialect required for FastAPI/Celery
  value = "postgresql+asyncpg://prism:${var.db_password}@${aws_db_instance.prism_postgres.endpoint}/prism"
}

# ── Outputs ──────────────────────────────────────────────────────────────
output "ec2_public_ip" {
  description = "Elastic IP — use this for GitHub webhook URL and CI SSH target"
  value       = aws_eip.prism_eip.public_ip
}

output "rds_endpoint" {
  description = "RDS endpoint (internal, not publicly accessible)"
  value       = aws_db_instance.prism_postgres.endpoint
}
