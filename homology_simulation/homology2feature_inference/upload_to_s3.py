#!/usr/bin/env python3
"""Upload entropy_maps.npy, label_matrices.npy, and metadata.csv to S3."""
from pathlib import Path

import boto3

from config import AWS_PROFILE_NAME

S3_BUCKET = "homology-experiment"
S3_PREFIX = "simulation_dataset"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "simulation_dataset"

session = boto3.Session(profile_name=AWS_PROFILE_NAME)
s3 = session.client("s3")

for fname in ("entropy_maps.npy", "label_matrices.npy", "metadata.csv"):
    local_path = OUTPUT_DIR / fname
    s3_key = f"{S3_PREFIX}/{fname}"
    print(f"Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)

print("Done.")
