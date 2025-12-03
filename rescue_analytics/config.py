# rescue_analytics/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

import os
import yaml
from dotenv import load_dotenv

# load .env khi chạy local
load_dotenv()

# Streamlit secrets (chỉ tồn tại khi chạy trên Streamlit)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def _get_conf(key: str, default: str = "") -> str:
    """
    Ưu tiên:
    1) st.secrets[key]  (Streamlit Cloud)
    2) os.getenv(key)   (local .env)
    """
    if HAS_STREAMLIT and hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)


@dataclass
class DBConfig:
    host: str = _get_conf("DB_HOST", "localhost")
    port: int = int(_get_conf("DB_PORT", "5432"))
    name: str = _get_conf("DB_NAME", "rescue_db")
    user: str = _get_conf("DB_USER", "rescue_user")
    password: str = _get_conf("DB_PASSWORD", "rescue_password")


@dataclass
class S3Config:
    endpoint_url: str = _get_conf("S3_ENDPOINT_URL", "")
    access_key: str = _get_conf("S3_ACCESS_KEY", "")
    secret_key: str = _get_conf("S3_SECRET_KEY", "")
    bucket: str = _get_conf("S3_BUCKET", "")
    region: str = _get_conf("S3_REGION", "eu-west-1")


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
        raw = yaml.safe_load(f) or {}

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
