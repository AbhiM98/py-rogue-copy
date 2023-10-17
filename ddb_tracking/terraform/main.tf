terraform { 
    backend "s3" {
        bucket = "rogues-tf-state"
        key = "terraform.tfstate"
        region = "us-east-1"
    }
    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = ">=4.52.0"
        }
    }
    required_version = "~> 1.0"
}

provider "aws" {
    region = "us-east-1"
}