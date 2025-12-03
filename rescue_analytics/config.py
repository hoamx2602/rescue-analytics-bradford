
# rescue_analytics/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import List

import os
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DBConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "rescue_db")
    user: str = os.getenv("DB_USER", "rescue_user")
    password: str = os.getenv("DB_PASSWORD", "rescue_password")


@dataclass
class S3Config:
    access_key: str = os.getenv("S3_ACCESS_KEY", "")
    secret_key: str = os.getenv("S3_SECRET_KEY", "")
    bucket: str = os.getenv("S3_BUCKET", "")
    region: str = os.getenv("S3_REGION", "eu-west-1")  # AWS region, ví dụ: eu-west-1, us-east-1


@dataclass
class DataSource:
    name: str
    prefix: str
    modality: str


@dataclass
class Settings:
    db: DBConfig
    s3: S3Config
    sources: List[DataSource]


def load_sources(config_path: Path = Path("config/data_sources.yml")) -> List[DataSource]:
    if not config_path.exists():
        return []
    
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []

    sources = []
    for src in raw.get("sources", []):
        sources.append(
            DataSource(
                name=src["name"],
                prefix=src["prefix"],
                modality=src.get("modality", "unknown"),
            )
        )
    return sources


settings = Settings(
    db=DBConfig(),
    s3=S3Config(),
    sources=load_sources(),
)
