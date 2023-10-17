# Variables
variable "base_name" {
    description = "Base name for ECS process and tasks."
    type = string 
}

variable "image_name" {
    description = "Variable part of docker image name, ie for 'rogues-prod-inference-img' this would be 'inference'."
    type = string 
}

variable "script_name" {
    description = "Name of the script to run."
    type = string
}

variable "script_args" {
    description = "Arguments to pass to the script."
    type = string
}

variable "entrypoint" {
    description = "Path to the script specified in 'script_name'."
    type = string 
}

variable "conda_env" {
    description = "Name of the conda environment to run the script in."
    type = string
    default = "rogue-venv"
}

variable "cmd_prefix" {
    description = "Command prefix to run the script."
    type = string
}

variable "cpu" {
    description = "Number of CPU units to allocate to the task."
    type = number
    default = 1024
}

variable "memory" {
    description = "Amount of memory to allocate to the task."
    type = number
    default = 2048
}

variable "gpu" {
    description = "Number of vGPUs to allocate to the task."
    type = string
    default = "0"
}

variable "log_group" {
    description = "Log group for the ECS process."
    type = object({
        name = string
        retention_in_days = number
    })
}

variable "exec_arn" {
    description = "Execution ARN for the ECS process."
    type = string
}

variable "ecs_cluster_id" {
    description = "ECS cluster ID."
    type = string
}

variable "ecs_cluster_name" {
    description = "ECS cluster name."
    type = string
}

variable "ecs_launch_type" { 
    description = "ECS launch type."
    type = string
    default = "FARGATE"
}

variable "subnets" {
    description = "List of subnet ids"
    type = list(string)
}

variable "securityGroups" {
    description = "Security group ids"
    type = list(string)
}


