# scripts/etl_extract_from_s3.py
from rescue_analytics.etl.extract_from_s3 import run_full_extract


if __name__ == "__main__":
    run_full_extract()
