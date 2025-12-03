# scripts/init_db.py
from sqlalchemy import text

from rescue_analytics.db import engine


DDL = """
CREATE TABLE IF NOT EXISTS images (
    id               BIGSERIAL PRIMARY KEY,
    s3_key           TEXT NOT NULL UNIQUE,
    source_name      TEXT,
    filename         TEXT,
    modality         VARCHAR(32),
    has_person       BOOLEAN,
    num_persons      INT,
    brightness       FLOAT,
    width            INT,
    height           INT,
    s3_etag          TEXT,
    s3_last_modified TIMESTAMP,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS yolo_boxes (
    id          BIGSERIAL PRIMARY KEY,
    image_id    BIGINT REFERENCES images(id) ON DELETE CASCADE,
    class_id    INT,
    x_center    FLOAT,
    y_center    FLOAT,
    box_width   FLOAT,
    box_height  FLOAT
);
"""


def init_db():
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    print("DB schema created/updated.")


if __name__ == "__main__":
    init_db()
