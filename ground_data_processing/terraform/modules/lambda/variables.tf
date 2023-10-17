# Variables 
variable "base_name" {
    description = "Base name. Seperate with dashes."
    type = string
}

variable "handler" {
    description = "Lambda handler function name."
    type = string
}

variable "role" {
    description = "Lambda execution role."
    type = string
}

variable "runtime" {
    description = "Runtime for the lambda function."
    default = "python3.9"
    type = string
}

variable "env_variables" { 
    description = "Environment variables for the lambda function."
    default = {"git" = "gud"}
    type = map(string)
}

variable "s3_data" {
    description = "S3 data for the lambda code."
    type = object({
        bucket = string
        key = string
        source_code_hash = string
    })
}

variable "timeout" {
    description = "Timeout for the lambda function."
    default = 60
    type = number
}

locals {
    lambda_name = "rogues-${terraform.workspace}-${var.base_name}"
}