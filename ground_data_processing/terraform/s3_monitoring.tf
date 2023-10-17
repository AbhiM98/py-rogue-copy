##################
# Adding S3 bucket as trigger to lambda
##################

resource "aws_s3_bucket" "sentera_rogues_data" {
    bucket = "sentera-rogues-data"
}

resource "aws_s3_bucket_notification" "aws_lambda_trigger" {
    bucket = "${aws_s3_bucket.sentera_rogues_data.id}"
    lambda_function {
        lambda_function_arn = "${aws_lambda_function.s3_monitoring_lambda.arn}"
        events              = ["s3:ObjectCreated:*"]
        filter_suffix       = ".mp4"
    }
}

resource "aws_lambda_permission" "s3_monitoring_permission" {
    statement_id  = "AllowS3Invoke"
    action        = "lambda:InvokeFunction"
    function_name = "${aws_lambda_function.s3_monitoring_lambda.arn}"
    principal = "s3.amazonaws.com"
    source_arn = "arn:aws:s3:::${aws_s3_bucket.sentera_rogues_data.id}"
}