# app/s3_io.py
import os, re, tempfile
import boto3
from botocore.config import Config

BUCKET = os.getenv("S3_BUCKET", "")
ENDPOINT = os.getenv("AWS_S3_ENDPOINT")  # MinIO면 http://localhost:9000
FORCE_PATH = os.getenv("AWS_S3_FORCE_PATH_STYLE", "true").lower() == "true"
REGION = os.getenv("AWS_REGION", "ap-northeast-2")

_cfg = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path" if FORCE_PATH else "auto"},
)

def _client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        config=_cfg,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def normalize_to_s3_uri(uri: str) -> str:
    """https://bucket.s3.amazonaws.com/key, https://bucket.s3.ap-northeast-2.amazonaws.com/key,
       http(s)://endpoint/bucket/key, s3://bucket/key -> s3://bucket/key"""
    if uri.startswith("s3://"):
        return uri
    m = re.match(r"https?://([^.]+)\.s3(?:[.-][^/]+)?\.amazonaws\.com(?:\.cn)?/(.+)", uri)
    if m:  # virtual-hosted-style
        bucket, key = m.group(1), m.group(2)
        return f"s3://{bucket}/{key}"
    m2 = re.match(r"https?://[^/]+/([^/]+)/(.+)", uri)
    if m2:  # path-style (MinIO/LocalStack)
        bucket, key = m2.group(1), m2.group(2)
        return f"s3://{bucket}/{key}"
    raise ValueError(f"unsupported S3 URL: {uri}")

def _split_s3(s3_uri: str):
    assert s3_uri.startswith("s3://")
    rest = s3_uri[5:]
    i = rest.find("/")
    if i < 0:
        raise ValueError(f"invalid s3 uri: {s3_uri}")
    return rest[:i], rest[i + 1 :]

def download_s3_to_temp(s3_or_https_uri: str) -> str:
    print(f"[s3_io] Downloading: {s3_or_https_uri}")

    try:
        s3_uri = normalize_to_s3_uri(s3_or_https_uri)
        bucket, key = _split_s3(s3_uri)

        s3 = _client()
        _, ext = os.path.splitext(key)
        fd, tmp = tempfile.mkstemp(suffix=ext or ".bin")
        os.close(fd)

        s3.download_file(bucket, key, tmp)
        file_size = os.path.getsize(tmp)
        print(f"[s3_io] ✅ Downloaded {file_size} bytes to {tmp}")

        return tmp

    except Exception as e:
        print(f"[s3_io] ❌ Download failed: {e}")
        raise

def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    if not BUCKET:
        raise RuntimeError("S3_BUCKET is not set")
    _client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"s3://{BUCKET}/{key}"
