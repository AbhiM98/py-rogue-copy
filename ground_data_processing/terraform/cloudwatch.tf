# LAMBDAS
resource "aws_cloudwatch_log_group" "s3_monitoring_log_group" {
    name = "/aws/lambda/${aws_lambda_function.s3_monitoring_lambda.function_name}"
    retention_in_days = 7
}

# ECS
resource "aws_cloudwatch_log_group" "ecs_api" {
    name = "${terraform.workspace}-rogues-ecs-api-log-group"
    retention_in_days = 7
}