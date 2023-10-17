# Repo: https://github.com/terraform-aws-modules/terraform-aws-ecs

locals {
  region = "us-east-1"
  name = "${terraform.workspace}-rogues-ecs-cluster"

  user_data = <<-EOT
    #!/bin/bash
    cat <<'EOF' >> /etc/ecs/ecs.config
    ECS_CLUSTER=${local.name}
    ECS_LOGLEVEL=debug
    ECS_ENABLE_GPU_SUPPORT=true
    ECS_ENABLE_CONTAINER_METADATA=true
    EOF
  EOT

  tags = {
    Name = local.name
  }

  min_size = 0
  max_size = 0 # We'll set this higher when an ecs lambda is called
}

################################################################################
# ECS Module
################################################################################

module "ecs" {
  source = "terraform-aws-modules/ecs/aws"
  version = "4.1.1"

  cluster_name = local.name

  cluster_configuration = {
    execute_command_configuration = {
      logging = "OVERRIDE"
      log_configuration = {
        # You can set a simple string and ECS will create the CloudWatch log group for you
        # or you can create the resource yourself as shown here to better manage retetion, tagging, etc.
        # Embedding it into the module is not trivial and therefore it is externalized
        cloud_watch_log_group_name = aws_cloudwatch_log_group.this.name
      }
    }
  }

  default_capacity_provider_use_fargate = false

  # Capacity provider - Fargate
  fargate_capacity_providers = {
    FARGATE      = {}
    FARGATE_SPOT = {}
  }

  # Capacity provider - autoscaling groups
  autoscaling_capacity_providers = {
    one = {
      name = local.name
      auto_scaling_group_arn         = module.autoscaling["one"].autoscaling_group_arn
      managed_termination_protection = "ENABLED"

      managed_scaling = {
        maximum_scaling_step_size = 5
        minimum_scaling_step_size = 1
        status                    = "ENABLED"
        target_capacity           = 80 # Percent usage
      }

      default_capacity_provider_strategy = {
        weight = 60
        base   = 20
      }
    }
  }

  tags = local.tags
}

################################################################################
# Supporting Resources
################################################################################

# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/retrieve-ecs-optimized_AMI.html
# use gpu optimized ami
data "aws_ssm_parameter" "ecs_optimized_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended"
}

module "autoscaling" {
  source  = "terraform-aws-modules/autoscaling/aws"
  version = "6.5.3"

  for_each = {
    one = {
      # instance_type = "p3.2xlarge" # same as rogues-dev-trev-3
      instance_type = "g3.4xlarge" # same as rogues-ffmpeg-processing
    }
  }

  name = "${local.name}-${each.key}"

  image_id      = jsondecode(data.aws_ssm_parameter.ecs_optimized_ami.value)["image_id"]
  instance_type = each.value.instance_type

  security_groups                 = [module.autoscaling_sg.security_group_id]
  user_data                       = base64encode(local.user_data)
  ignore_desired_capacity_changes = true

  create_iam_instance_profile = true
  iam_role_name               = local.name
  iam_role_description        = "ECS role for ${local.name}"
  iam_role_policies = {
    AmazonEC2ContainerServiceforEC2Role = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
    AmazonSSMManagedInstanceCore        = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  # Hard-coded subnets are from mfstand, we reuse them since vpc's are limited
  vpc_zone_identifier = ["subnet-08f7251ac284d8aef","subnet-0b8cc8e605273eceb","subnet-0ab2d8e482ae18c84"]
  health_check_type   = "EC2"
  min_size            = local.min_size
  max_size            = local.max_size
  desired_capacity    = 0 # Gets overridden by ECS

  # https://github.com/hashicorp/terraform-provider-aws/issues/12582
  autoscaling_group_tags = {
    AmazonECSManaged = true
  }

  # Required for  managed_termination_protection = "ENABLED"
  protect_from_scale_in = true

  tags = local.tags

  # https://github.com/terraform-aws-modules/terraform-aws-autoscaling
  # Configure EBS Storage volumne
  block_device_mappings = [
    {
      # Root volume
      device_name = "/dev/xvda"
      no_device   = 0
      ebs = {
        delete_on_termination = true
        encrypted             = true
        volume_size           = 3000 # GB
        volume_type           = "gp2"
      }
    }
  ]

  # Schedule scaling actions
  # create_schedule = true
  # schedules = {
  #   # Set desired capacity to 0 on weekends
  #   "weekend" = {
  #     desired_capacity = 0
  #     # [Minute] [Hour] [Day_of_Month] [Month_of_Year] [Day_of_Week]
  #     # Cron examples: https://crontab.guru/examples.html
  #     recurrence       = "0 0 * * 5-6" # 12am on Friday and Saturday
  #   }
  #   # Set max capacity to max_size on weekdays
  #   "weekday" = {
  #     max_size    = local.max_size
  #     recurrence  = "0 0 * * 1-4" # 12am on Monday through Thursday
  #   }
  # }
  
}

module "autoscaling_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "4.16.0"

  name        = local.name
  description = "Autoscaling group security group"
  # Same as mfstand
  vpc_id      = "vpc-056ce2689bad9a608"

  ingress_cidr_blocks = ["0.0.0.0/0"]
  ingress_rules       = ["https-443-tcp"]

  egress_rules = ["all-all"]

  tags = local.tags
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.18.1"

  # VPC exists on mfstand, so we dont' need to create it
  create_vpc = false

  name = local.name
  cidr = "10.99.0.0/18"

  azs             = ["${local.region}a", "${local.region}b", "${local.region}c"]
  public_subnets  = ["10.99.0.0/24", "10.99.1.0/24", "10.99.2.0/24"]
  private_subnets = ["10.99.3.0/24", "10.99.4.0/24", "10.99.5.0/24"]

  enable_nat_gateway      = true
  single_nat_gateway      = true
  enable_dns_hostnames    = true
  map_public_ip_on_launch = false

  # Use same eip for all production envs
  reuse_nat_ips       = true
  external_nat_ip_ids = ["eipalloc-0985061672b948b35"]

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/ecs/${local.name}"
  retention_in_days = 7

  tags = local.tags
}