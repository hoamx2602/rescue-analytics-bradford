# rescue_analytics/analysis/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from rescue_analytics.db import engine


@dataclass
class EmbeddingResult:
    df: pd.DataFrame  # dataframe gốc + cột embedding
    cols_used: List[str]
    method: str       # "pca" hoặc "tsne"


def load_feature_dataframe(
    limit: Optional[int] = None,
    only_has_person: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load dữ liệu ảnh từ DB và xây feature tabular.
    Bạn có thể bổ sung thêm feature (ví dụ stats từ bounding boxes) nếu muốn.
    """
    query = """
        SELECT
            i.id,
            i.s3_key,
            i.source_name,
            i.filename,
            i.modality,
            i.has_person,
            i.num_persons,
            i.brightness,
            i.width,
            i.height
        FROM images i
    """

    if only_has_person is True:
        query += " WHERE i.has_person = TRUE"
    elif only_has_person is False:
        query += " WHERE i.has_person = FALSE"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn)

    # Feature engineering cơ bản
    df["aspect_ratio"] = df["width"] / df["height"]
    df["modality_code"] = df["modality"].astype("category").cat.codes

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Từ df → matrix X để đưa vào PCA/t-SNE.
    """
    feature_cols = [
        "brightness",
        "width",
        "height",
        "aspect_ratio",
        "num_persons",
        "modality_code",
    ]

    X = df[feature_cols].astype(float).values
    return X, feature_cols


def perform_pca(
    limit: Optional[int] = 2000,
    only_has_person: Optional[bool] = None,
    n_components: int = 2,
) -> EmbeddingResult:
    """
    PCA 2D cho visualization.
    """
    df = load_feature_dataframe(limit=limit, only_has_person=only_has_person)
    if df.empty:
        raise ValueError("No data loaded from DB for PCA.")

    X, feature_cols = build_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    emb = pca.fit_transform(X_scaled)

    df["pca_1"] = emb[:, 0]
    df["pca_2"] = emb[:, 1]

    return EmbeddingResult(df=df, cols_used=feature_cols, method="pca")


def perform_tsne(
    limit: Optional[int] = 1500,
    only_has_person: Optional[bool] = None,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> EmbeddingResult:
    """
    t-SNE 2D cho visualization. Nên limit số điểm để tránh quá chậm.
    """
    df = load_feature_dataframe(limit=limit, only_has_person=only_has_person)
    if df.empty:
        raise ValueError("No data loaded from DB for t-SNE.")

    X, feature_cols = build_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(X_scaled)

    df["tsne_1"] = emb[:, 0]
    df["tsne_2"] = emb[:, 1]

    return EmbeddingResult(df=df, cols_used=feature_cols, method="tsne")


def simple_feature_stats() -> pd.DataFrame:
    """
    Feature selection nhẹ: xem variance, min, max, corr…
    Dùng cho phần 'feature analysis' trong báo cáo.
    """
    df = load_feature_dataframe(limit=None)
    if df.empty:
        raise ValueError("No data loaded from DB.")

    numeric_cols = ["brightness", "width", "height", "aspect_ratio", "num_persons"]
    desc = df[numeric_cols].describe().T
    desc["variance"] = df[numeric_cols].var()
    return desc
