import boto3
import json
import threading
import time
from logger import debug_logger

class S3ConfigManager:
    def __init__(self, bucket_name, config_key, aws_region=None):
        self.bucket_name = bucket_name
        self.config_key = config_key
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.config_data = {}
        self.debug_logger = debug_logger()
        self.lock = threading.Lock()

    def fetch_config(self):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.config_key)
            content = response['Body'].read().decode('utf-8')
            config = json.loads(content)
            with self.lock:
                self.config_data = config
            debug_logger().debug(f"Config fetched from S3: {config}")
        except Exception as e:
            debug_logger().exception(f"Error fetching config from S3: {e}")

    def get_config(self):
        with self.lock:
            return self.config_data

    def start_periodic_refresh(self, interval_seconds=3600):
        def refresh_loop():
            time.sleep(interval_seconds)  # Wait for the specified interval before the first fetch
            while True:
                self.fetch_config()
                time.sleep(interval_seconds)
        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()