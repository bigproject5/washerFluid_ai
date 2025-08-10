import os, tempfile
import boto3

BUCKET = os.getenv("S3_BUCKET")


def download_s3_to_temp(s3_uri: str) -> str:
    assert s3_uri.startswith("s3://"), "s3_uri must start with s3://"
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    s3 = boto3.client("s3")
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(key)[1])
    os.close(fd)
    s3.download_file(bucket, key, tmp)
    return tmp


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    assert BUCKET, "S3_BUCKET is not set"
    boto3.client("s3").put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"s3://{BUCKET}/{key}"