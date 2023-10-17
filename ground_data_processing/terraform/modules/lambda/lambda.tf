# Lambda function
resource "aws_lambda_function" "lambda_function" {
    function_name = local.lambda_name
    
    s3_bucket = var.s3_data.bucket
    s3_key = var.s3_data.key

    runtime = var.runtime 
    handler = var.handler

    source_code_hash = var.s3_data.source_code_hash 
    environment {
        variables = var.env_variables
    }
    
    role = var.role 
    timeout = var.timeout
}

# Cloudwatch log group
resource "aws_cloudwatch_log_group" "lambda_log_group" {
    name = "/aws/lambda/${local.lambda_name}"
    retention_in_days = 7
}