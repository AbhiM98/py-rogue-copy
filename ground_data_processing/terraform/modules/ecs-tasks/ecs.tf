locals {
    gpu_def = jsonencode([
        {
            "name": "${terraform.workspace}-rogues-${var.base_name}-task",
            "image": "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-${terraform.workspace}-${var.image_name}-img",
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-region": "us-east-1",
                    "awslogs-group": "${var.log_group.name}",
                    "awslogs-stream-prefix": "rogues-${var.base_name}-task"
                }
            },
            "entryPoint": [
                # "sh",
                # "-c",
                # "nvidia-smi",
                "/bin/bash",
                "-c",
                "conda run --no-capture-output -n ${var.conda_env} ${var.cmd_prefix}python ${var.entrypoint}${var.script_name} ${var.script_args}"
            ],
            "resourceRequirements": [
                {
                    "type" : "GPU",
                    "value": "${var.gpu}"
                }
            ],
        }
    ])
    cpu_def = jsonencode([
        {
            "name": "${terraform.workspace}-rogues-${var.base_name}-task",
            "image": "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-${terraform.workspace}-${var.image_name}-img",
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-region": "us-east-1",
                    "awslogs-group": "${var.log_group.name}",
                    "awslogs-stream-prefix": "rogues-${var.base_name}-task"
                }
            },
            "entryPoint": [
                "/bin/bash",
                "-c",
                "conda run --no-capture-output -n ${var.conda_env} ${var.cmd_prefix}python ${var.entrypoint}${var.script_name} ${var.script_args}"
            ],
        }
    ])
    # Choose the appropriate container definition based on whether or not a GPU is requested
    container_defs = var.gpu == "0" ? local.cpu_def : local.gpu_def
}

# Task definition 
resource "aws_ecs_task_definition" "task_definition" {
    family = "${terraform.workspace}-rogues-${var.base_name}-task-definition"
    container_definitions = local.container_defs
    # Autoscaling group already has the correct permissions, so we don't need to attach a policy here for EC2 jobs
    task_role_arn = var.exec_arn
    execution_role_arn = var.exec_arn
    cpu = var.cpu
    memory = var.memory
    requires_compatibilities = ["FARGATE"]
    network_mode = "awsvpc"
}


# Associated Service 
resource "aws_ecs_service" "service" {
    name = "${terraform.workspace}-rogues-${var.base_name}-service"
    cluster = var.ecs_cluster_id
    task_definition = aws_ecs_task_definition.task_definition.arn
    desired_count = 0
    launch_type = "${var.ecs_launch_type}"

    network_configuration {
        assign_public_ip = "${var.ecs_launch_type}" == "FARGATE" ? true : false

        security_groups = var.securityGroups

        subnets = var.subnets
    }
}