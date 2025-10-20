variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "ap-south-1"
}

variable "app_name" {
  description = "Name of the application"
  type        = string
  default     = "healthai"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

variable "ecr_repository" {
  description = "Name of the ECR repository"
  type        = string
  default     = "healthai"
}

variable "container_name" {
  description = "Name of the container"
  type        = string
  default     = "healthai"
}

variable "app_port" {
  description = "Port exposed by the docker image"
  type        = number
  default     = 8000
}

variable "app_count" {
  description = "Number of docker containers to run"
  type        = number
  default     = 2
}

variable "fargate_cpu" {
  description = "Fargate instance CPU units to provision"
  type        = number
  default     = 1024
}

variable "fargate_memory" {
  description = "Fargate instance memory to provision"
  type        = number
  default     = 2048
}

variable "ecr_image" {
  description = "ECR image URI"
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "List of private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnets" {
  description = "List of public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "azs" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b"]
}
