# HealthAI Deployment

This directory contains the Terraform configuration for deploying the HealthAI application to AWS.

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured with access keys
3. Terraform (>= 1.0.0)
4. Docker
5. GitHub repository with GitHub Actions enabled

## Setup Instructions

### 1. Configure AWS Credentials

Create an IAM user with the following permissions:
- AmazonEC2ContainerRegistryFullAccess
- AmazonECS_FullAccess
- IAMFullAccess
- AmazonVPCFullAccess
- CloudWatchLogsFullAccess
- ElasticLoadBalancingFullAccess

### 2. Set up GitHub Secrets

Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: (Optional, defaults to ap-south-1)

### 3. Deploy Infrastructure

The infrastructure will be automatically deployed when you push to the `main` branch. The GitHub Actions workflow will:
1. Build and push the Docker image to ECR
2. Initialize and apply the Terraform configuration
3. Deploy the application to ECS Fargate

### 4. Access the Application

After deployment, you can access the application at the URL provided in the GitHub Actions output (look for the `load_balancer_url` output).

## Manual Deployment (Optional)

If you need to deploy manually:

```bash
# Initialize Terraform
cd terraform
terraform init

# Plan the deployment
terraform plan -var="image_tag=latest"

# Apply the configuration
terraform apply -var="image_tag=latest"
```

## Architecture

The deployment includes:
- ECS Fargate for container orchestration
- Application Load Balancer for traffic distribution
- ECR for Docker image storage
- VPC with public and private subnets
- CloudWatch Logs for monitoring

## Variables

Key variables you might want to customize:
- `app_name`: Name of the application (default: "healthai")
- `environment`: Deployment environment (default: "prod")
- `app_port`: Port exposed by the application (default: 8000)
- `fargate_cpu`: CPU units for Fargate tasks (default: 1024)
- `fargate_memory`: Memory for Fargate tasks in MB (default: 2048)

## Cleanup

To destroy all resources:

```bash
cd terraform
terraform destroy
```

## Monitoring

- **CloudWatch Logs**: Application logs are sent to CloudWatch Logs
- **ECS Service**: Monitor the ECS service in the AWS Management Console
- **Load Balancer**: Check the ALB metrics in the EC2 console
