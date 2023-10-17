# S3 Monitoring Lambda
resource "aws_lambda_function" "s3_monitoring_lambda" {
    function_name    = "rogues-${terraform.workspace}-s3-monitoring"
    s3_bucket        = aws_s3_bucket.lambda_s3_bucket.id
    s3_key           = aws_s3_object.lambda_zip.key
    role             = aws_iam_role.rogues_lambda_exec_role.arn 
    handler          = "s3_monitoring.s3_monitoring"
    runtime          = "python3.9"
    timeout          = 60
    source_code_hash = data.archive_file.lambda_zip.output_base64sha256
}

locals {
    lambda_defs = [
        # Processing Steps
        {
            base_name = "generate-ds-splits"
            handler = "ecs_procs.generate_ds_splits"
            env_variables = module.ecs_tasks["generate-ds-splits"]
            timeout = 900 # 15 min, max timeout
        },
        {
            base_name = "extract-frames"
            handler = "ecs_procs.extract_frames"
            env_variables = module.ecs_tasks["extract-frames"]
            timeout = 900 # 15 min, max timeout
        },
        {
            base_name = "run-all"
            handler = "ecs_procs.run_all"
            env_variables = module.ecs_tasks["run-all"]
            timeout = 900 # 15 min, max timeout
        },
        {
            base_name = "run-paddle-inference"
            handler = "ecs_procs.run_paddle_inference"
            env_variables = module.ecs_tasks["run-paddle-inference"]
            timeout = 900 # 15 min, max timeout
        },
        {
            base_name = "prep-and-run-paddle-inference"
            handler = "ecs_procs.prep_and_run_paddle_inference"
            env_variables = module.ecs_tasks["prep-and-run-paddle-inference"]
            timeout = 900 # 15 min, max timeout
        },
        # Batch jobs
        {
            base_name = "batch-run-processing_step"
            handler = "ecs_procs.batch_run_processing_step"
            timeout = 900 # 15 min, max timeout
        },
        {
            base_name = "batch-prep-and-run-paddle-inference"
            handler = "ecs_procs.batch_prep_and_run_paddle_inference"
            timeout = 900 # 15 min, max timeout
        },
        # Reports
        {
            base_name = "write-report-csv"
            handler = "grd_reports.write_report_csv"
        },
        # SNS
        {
            base_name = "send-sns-message-to-name"
            handler = "sns.send_sns_message_to_name"
        }
    ]
}

module "lambdas" {
    source = "./modules/lambda"

    for_each = {
        for index, lambda_def in local.lambda_defs:
        lambda_def.base_name => lambda_def
    }
    base_name = each.value.base_name
    handler = each.value.handler
    env_variables = try(each.value.env_variables, {"git": "gud"})
    role = aws_iam_role.rogues_lambda_exec_role.arn
    timeout = try(each.value.timeout, 60)
    s3_data = {
        bucket = aws_s3_bucket.lambda_s3_bucket.id
        key = aws_s3_object.lambda_zip.key
        source_code_hash = data.archive_file.lambda_zip.output_base64sha256
    }
}