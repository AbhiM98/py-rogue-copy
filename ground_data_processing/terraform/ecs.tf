# ECS Tasks 
# For valid cpu+memory options: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
# 1 GB memory = 1024

# g3.4xlarge total resources
# cpu = 16384, 16 CPU units
# memory = 122852, 112 GB

locals {
    image_processing_entrypoint = "/app/ground_data_processing/processing_entrypoints/"
    inferencing_entrypoint = "/app/lambdas/"
    image_processing_args = "--field_name \"$FIELD_NAME\" --date \"$DATE\" --row_number \"$ROW_NUMBER\" --ds_split_numbers \"$DS_SPLIT_NUMBERS\" --nadir_crop_height \"$NADIR_CROP_HEIGHT\" --nadir_crop_width \"$NADIR_CROP_WIDTH\" --overwrite \"$OVERWRITE\" --rerun \"$RERUN\""
    ecs_tasks = [
        {
            "base_name" = "generate-ds-splits",
            "script_name" = "generate_ds_splits.py",
            "script_args" = local.image_processing_args,
            cpu = 8192, # 8 CPU units
            memory = 49152, # 48 GB, this allows it to fit on an instance w/4 inference tasks
        },
        {
            "base_name" = "extract-frames",
            "script_name" = "extract_frames.py",
            "script_args" = local.image_processing_args,
            cpu = 8192, # 8 CPU units
            memory = 49152, # 48 GB, this allows it to fit on an instance w/4 inference tasks
        },
        {
            "base_name" = "run-all",
            "script_name" = "run_all.py",
            "script_args" = local.image_processing_args,
            # Enough for 4 concurrent tasks
            # cpu = 4096, # 4 CPU units
            # memory = 28672, # 28 GB
            
            # Enough for 2 concurrent tasks
            cpu = 8192, # 8 CPU units
            memory = 49152, # 48 GB, this allows it to fit on an instance w/4 inference tasks
            # memory = 46080, # 45 GB, EC2 only
        },
        {
            "base_name" = "prep-and-run-paddle-inference",
            "script_name" = "prep_and_run_paddle_inference.py",
            "script_args" = local.image_processing_args,
            
            # Enough for 6 concurrent tasks
            cpu = 2048 # 2 CPU units
            memory = 16384  # 16 GB
            # memory = 19115 # 18.7 GB, EC2 only
            # ecs_launch_type = "FARGATE"
        },
        {
            "base_name" = "run-paddle-inference",
            "script_name" = "run_paddle_inference.py",
            "script_args" = local.image_processing_args,
            "entrypoint" = local.inferencing_entrypoint,
            "image_name" = "inference"
            "conda_env" = "paddle-venv"
            # Enough for 4 concurrent tasks + FARGATE compatibility
            cpu = 4096 # 4 CPU units
            memory = 28672  # 28 GB

            # Enough for 6 concurrent tasks + FARGATE compatibility
            # cpu = 2048 # 2 CPU units
            # memory = 16384  # 16 GB
            # memory = 19115 # 18.7 GB, EC2 only
            # gpu = "1" # 1 vGPU
        },
    ]
}

module "ecs_tasks" {
    source = "./modules/ecs-tasks"
    for_each = {
        for index, ecs_task in local.ecs_tasks:
        ecs_task.base_name => ecs_task
    }
    base_name = each.value.base_name 
    image_name = try(each.value.image_name, "ecs")
    script_name = each.value.script_name
    script_args = each.value.script_args
    entrypoint = try(each.value.entrypoint, local.image_processing_entrypoint)
    conda_env = try(each.value.conda_env, "rogue-venv")
    cmd_prefix = try(each.value.cmd_prefix, "")
    cpu = try(each.value.cpu, 1024)  // default to 1 CPU unit
    memory = try(each.value.memory, 2048)  // default to 2 GB of memory
    gpu = try(each.value.gpu, "0")  // default to 0 vGPUs
    ecs_launch_type = try(each.value.ecs_launch_type, "EC2")

    # Hard-coded subnets are from mfstand, we reuse them since vpc's are limited
    subnets = ["subnet-08f7251ac284d8aef","subnet-0b8cc8e605273eceb","subnet-0ab2d8e482ae18c84"]
    securityGroups = [module.autoscaling_sg.security_group_id]

    log_group = aws_cloudwatch_log_group.ecs_api
    exec_arn = aws_iam_role.rogues_ecs_task_exec_role.arn
    ecs_cluster_id = module.ecs.cluster_id
    ecs_cluster_name = module.ecs.cluster_name
}