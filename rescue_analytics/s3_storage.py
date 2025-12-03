
# rescue_analytics/s3_storage.py
import boto3

from .config import settings


def get_s3_client():
    """Create and return a configured boto3 S3 client for AWS S3.
    
    Chỉ hỗ trợ AWS S3 thuần. Boto3 tự động tạo endpoint dựa trên region.
    """
    cfg = settings.s3
    
    # Xử lý trường hợp S3_REGION bị set thành "region" (invalid) hoặc empty
    region = cfg.region if cfg.region and cfg.region != "region" else "eu-west-1"
    
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
    )
