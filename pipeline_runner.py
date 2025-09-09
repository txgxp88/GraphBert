import torch
import os
from google.cloud import storage

def save_obj(obj, path):
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open("/tmp/tmp_save.pth", "wb") as f:
            torch.save(obj, f)
        blob.upload_from_filename("/tmp/tmp_save.pth")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)

def load_obj(path):
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        tmp_path = "/tmp/tmp_load.pth"
        blob.download_to_filename(tmp_path)
        return torch.load(tmp_path)
    else:
        return torch.load(path)
