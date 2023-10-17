# IAM Policy Documents 
data "aws_iam_policy_document" "ecs_task_assume_role" {
    statement {
        actions = ["sts:AssumeRole"]

        principals {
            type = "Service"
            identifiers = ["ecs-tasks.amazonaws.com"]
        }
    }
}

# IAM Policies 
resource "aws_iam_policy" "rogues_ecs_run_policy" {
    name = "rogues-${terraform.workspace}-ecs-run-policy"
    description = "Rogues ECS Run Policy"

    policy = jsonencode({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": [
                    "iam:*",
                    "ecs:RunTask",
                    "s3:*",
                    "dynamodb:*",
                    "ecs:*",
                    "ecr:*",
                    "ssm:*",
                ],
                "Effect": "Allow",
                "Resource": "*"
            }
        ]
    })
}

data "aws_iam_policy" "ecs_task_execution_role" {
    arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}


# IAM Roles
resource "aws_iam_role" "rogues_lambda_exec_role" {
    name = "rogues-${terraform.workspace}-lambda-exec-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Sid    = ""
                Principal = {
                    Service = "lambda.amazonaws.com"
                }
            }
        ]
    })
}
resource "aws_iam_role" "rogues_ecs_task_exec_role" {
    name = "rogues-${terraform.workspace}-ecs-task-exec-role"
    assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

# ECS Role Policy Attachments
# NOTE: I've manually added the following roles to prod-rogues-ecs-cluster-20230530192025938300000001
# AmazonEC2ContainerServiceforEC2Role, AmazonDynamoDBFullAccess, AmazonS3FullAccess, AWSLambdaRole, AmazonSNSFullAccess, AutoScalingFullAccess, AmazonECS_FullAccess
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_1" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = aws_iam_policy.rogues_ecs_run_policy.arn
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_2" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_3" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaRole"
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_4" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/AmazonSNSFullAccess"
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_5" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/AutoScalingFullAccess"
}
resource "aws_iam_role_policy_attachment" "ecs_task_exec_role_policy_attachment_6" {
    role = aws_iam_role.rogues_ecs_task_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# Lambda Role Policy Attachments
resource "aws_iam_role_policy_attachment" "_" {
    role = aws_iam_role.rogues_lambda_exec_role.id
    policy_arn = aws_iam_policy.rogues_ecs_run_policy.arn
}
resource "aws_iam_role_policy_attachment" "_2" {
    role = aws_iam_role.rogues_lambda_exec_role.id
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
resource "aws_iam_role_policy_attachment" "_3" {
    role = aws_iam_role.rogues_lambda_exec_role.name 
    policy_arn = data.aws_iam_policy.ecs_task_execution_role.arn
}
resource "aws_iam_role_policy_attachment" "_4" {
    role = aws_iam_role.rogues_lambda_exec_role.name 
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaRole"
}
resource "aws_iam_role_policy_attachment" "_5" {
    role = aws_iam_role.rogues_lambda_exec_role.name 
    policy_arn = "arn:aws:iam::aws:policy/AmazonSNSFullAccess"
}
resource "aws_iam_role_policy_attachment" "_6" {
    role = aws_iam_role.rogues_lambda_exec_role.name 
    policy_arn = "arn:aws:iam::aws:policy/AutoScalingFullAccess"
}

