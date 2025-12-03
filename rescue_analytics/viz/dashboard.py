# rescue_analytics/viz/dashboard.py
import sys
from pathlib import Path

# Add project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from rescue_analytics.db import engine
from rescue_analytics.s3_storage import get_presigned_url
from rescue_analytics.analysis.embeddings import (
    perform_pca,
    perform_tsne,
    simple_feature_stats,
)

from rescue_analytics.viz.draw_boxes import draw_boxes_on_image, load_image_from_s3


def load_main_dataframe(limit: int = 5000) -> pd.DataFrame:
    query = f"""
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
        ORDER BY i.id
        LIMIT {int(limit)}
    """

    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn)
    return df


def load_yolo_boxes_for_image(image_id: int) -> pd.DataFrame:
    query = """
        SELECT
            y.class_id,
            y.x_center,
            y.y_center,
            y.box_width,
            y.box_height
        FROM yolo_boxes y
        WHERE y.image_id = :image_id
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn, params={"image_id": image_id})
    return df


def page_overview():
    st.header("Overview")

    df = load_main_dataframe(limit=10000)
    if df.empty:
        st.warning("No data in images table yet.")
        return

    st.subheader("Basic stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total images", len(df))
    col2.metric("With person", int(df["has_person"].sum()))
    col3.metric("Without person", int((~df["has_person"]).sum()))

    st.write("### Modality breakdown")
    mod_counts = df["modality"].value_counts().reset_index()
    mod_counts.columns = ["modality", "count"]
    fig_mod = px.bar(mod_counts, x="modality", y="count", title="Images per modality")
    st.plotly_chart(fig_mod, use_container_width=True)

    st.write("### Brightness distribution")
    fig_bright = px.histogram(
        df,
        x="brightness",
        color="modality",
        nbins=40,
        title="Brightness distribution by modality",
    )
    st.plotly_chart(fig_bright, use_container_width=True)

    st.write("### Persons per image")
    fig_person = px.histogram(
        df,
        x="num_persons",
        color="modality",
        title="Number of persons per image",
    )
    st.plotly_chart(fig_person, use_container_width=True)


def page_embeddings():
    st.header("Embedding Visualization (PCA & t-SNE)")

    method = st.radio("Method", ["PCA", "t-SNE"], horizontal=True)
    only_has_person_opt = st.selectbox(
        "Filter by has_person",
        options=["All", "Only has person", "Only no person"],
    )

    only_has_person = None
    if only_has_person_opt == "Only has person":
        only_has_person = True
    elif only_has_person_opt == "Only no person":
        only_has_person = False

    max_points = st.slider("Max points", min_value=200, max_value=5000, value=1500, step=100)

    if st.button("Run embedding"):
        with st.spinner("Computing embedding..."):
            if method == "PCA":
                emb = perform_pca(limit=max_points, only_has_person=only_has_person)
                df_emb = emb.df
                fig = px.scatter(
                    df_emb,
                    x="pca_1",
                    y="pca_2",
                    color="modality",
                    symbol="has_person",
                    hover_data=["id", "source_name", "filename", "num_persons"],
                    title="PCA 2D embedding",
                )
            else:
                emb = perform_tsne(limit=max_points, only_has_person=only_has_person)
                df_emb = emb.df
                fig = px.scatter(
                    df_emb,
                    x="tsne_1",
                    y="tsne_2",
                    color="modality",
                    symbol="has_person",
                    hover_data=["id", "source_name", "filename", "num_persons"],
                    title="t-SNE 2D embedding",
                )

        st.plotly_chart(fig, use_container_width=True)

        st.write("**Feature statistics (for feature selection discussion)**")
        stats_df = simple_feature_stats()
        st.dataframe(stats_df)


def draw_yolo_boxes_placeholder():
    st.info(
        "Ở đây bạn có thể vẽ bounding boxes lên ảnh bằng PIL/Plotly nếu muốn. "
        "Hiện tại demo chỉ hiển thị bảng toạ độ."
    )


def page_image_explorer():
    st.header("Image Explorer")

    df = load_main_dataframe(limit=10000)
    if df.empty:
        st.warning("No data in images table yet.")
        return

    # Filters
    st.sidebar.subheader("Filters")
    modalities = ["All"] + sorted(df["modality"].dropna().unique().tolist())
    selected_mod = st.sidebar.selectbox("Modality", modalities)

    srcs = ["All"] + sorted(df["source_name"].dropna().unique().tolist())
    selected_src = st.sidebar.selectbox("Source", srcs)

    has_person_opt = st.sidebar.selectbox(
        "Has person",
        ["All", "Only has person", "Only no person"],
    )

    filtered = df.copy()
    if selected_mod != "All":
        filtered = filtered[filtered["modality"] == selected_mod]
    if selected_src != "All":
        filtered = filtered[filtered["source_name"] == selected_src]
    if has_person_opt == "Only has person":
        filtered = filtered[filtered["has_person"] == True]
    elif has_person_opt == "Only no person":
        filtered = filtered[filtered["has_person"] == False]

    st.write(f"{len(filtered)} images match filters.")

    # Chọn 1 ảnh để xem chi tiết
    image_ids = filtered["id"].tolist()
    if not image_ids:
        st.warning("No images after filtering.")
        return

    selected_id = st.selectbox("Select image id", image_ids)
    row = filtered[filtered["id"] == selected_id].iloc[0]

    col_left, col_right = st.columns([2, 1])


    with col_left:
        st.subheader("Image with bounding boxes")

        img = load_image_from_s3(row["s3_key"])
        boxes_df = load_yolo_boxes_for_image(int(row["id"]))
        boxes = boxes_df.to_dict(orient="records")

        if len(boxes) > 0:
            fig = draw_boxes_on_image(img, boxes)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image(img, caption=row["filename"], use_column_width=True)

    with col_right:
        st.subheader("Metadata")
        st.write(
            {
                "id": int(row["id"]),
                "source_name": row["source_name"],
                "filename": row["filename"],
                "modality": row["modality"],
                "has_person": bool(row["has_person"]),
                "num_persons": int(row["num_persons"]),
                "brightness": float(row["brightness"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )

        st.subheader("YOLO boxes")
        boxes_df = load_yolo_boxes_for_image(int(row["id"]))
        if boxes_df.empty:
            st.write("No YOLO boxes for this image.")
        else:
            st.dataframe(boxes_df)
            draw_yolo_boxes_placeholder()


def main():
    st.set_page_config(
        page_title="Rescue Visual Analytics",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Embeddings (PCA/t-SNE)", "Image Explorer"],
    )

    if page == "Overview":
        page_overview()
    elif page == "Embeddings (PCA/t-SNE)":
        page_embeddings()
    elif page == "Image Explorer":
        page_image_explorer()


if __name__ == "__main__":
    main()
