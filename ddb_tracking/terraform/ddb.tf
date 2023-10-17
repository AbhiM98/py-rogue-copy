resource "aws_dynamodb_table" "ground-rogues-data" {
    name = "ground-rogues-data"
    billing_mode = "PROVISIONED"
    read_capacity = 5
    write_capacity = 5
    hash_key = "field_name"
    range_key = "date#row_number"

    attribute { 
        name = "field_name"
        type = "S"
    }
    attribute { 
        name = "date#row_number"
        type = "S"
    }

    # attribute { 
    #     name = "row_number"
    #     type = "N"
    # }

    # attribute { 
    #     name = "date"
    #     type = "S"
    # }

    tags = { 
        Name = "Ground Rogues Data"
    }
}