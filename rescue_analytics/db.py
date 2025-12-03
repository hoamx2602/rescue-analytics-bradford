# rescue_analytics/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .config import settings


def get_db_url() -> str:
    cfg = settings.db
    return f"postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.name}"


engine = create_engine(get_db_url(), echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("DB connection OK:", result.scalar())
