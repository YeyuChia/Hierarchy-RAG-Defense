# src/adapters/iam/aws.py
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError
from .base import AccessControlBase

class AWSAccessControl(AccessControlBase):
    def __init__(self, user_credential: Dict[str, Any]):
        super().__init__(user_credential)
        self.s3 = self._client()

    def _client(self):
        ak = self.user_credential.get("aws_access_key")
        sk = self.user_credential.get("aws_secret_key")
        if not (ak and sk):
            return None
        try:
            return boto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk).client("s3")
        except Exception as e:
            print(f"[AWS][ERROR] Failed to create S3 client: {e}")
            return None

    def check_access(self, resource_metadata: Dict[str, Any]) -> bool:
        if not self.s3:
            return False
        bucket = (resource_metadata.get("parameters") or {}).get("bucket")
        key    = resource_metadata.get("file_name")
        if not (bucket and key):
            return False
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
        except Exception:
            return False
