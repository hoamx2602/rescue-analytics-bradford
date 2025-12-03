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
from rescue_analytics.config import settings
from rescue_analytics.analysis.embeddings import (
    perform_pca,
    perform_tsne,
    simple_feature_stats,
)
from rescue_analytics.viz.draw_boxes import (
    load_image_from_s3,
    draw_boxes_on_image,
)

# =========================
# Helpers: DB access
# =========================

def load_images(limit=None) -> pd.DataFrame:
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
        ORDER BY i.id
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"

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


def load_person_counts_by_source() -> pd.DataFrame:
    query = """
        SELECT
            i.source_name,
            COUNT(*) AS num_images,
            SUM(i.num_persons) AS total_persons,
            SUM(CASE WHEN i.has_person THEN 1 ELSE 0 END) AS images_with_person
        FROM images i
        GROUP BY i.source_name
        ORDER BY total_persons DESC
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn)
    return df


def load_total_person_boxes() -> int:
    query = "SELECT COUNT(*) FROM yolo_boxes"
    with engine.begin() as conn:
        n = conn.execute(text(query)).scalar()
    return int(n or 0)


# =========================
# Helpers: pairing IR / VIS
# =========================

def build_pair_key_from_filename(filename: str) -> str:
    """
    Dựa vào format:
      210417_MtErie_Enterprise_VIS_0003_00000002.jpeg
      210417_MtErie_Enterprise_IR_0004_00000002.jpeg

    -> stem = 210417_MtErie_Enterprise_VIS_0003_00000002
    -> parts = [210417, MtErie, Enterprise, VIS, 0003, 00000002]
    -> pair_key = '210417_MtErie_Enterprise_00000002'
    """
    name = filename.split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 6:
        # fallback: nếu format lạ, dùng luôn stem
        return stem

    date_part = parts[0]
    loc1 = parts[1]
    loc2 = parts[2]
    frame_id = parts[-1]  # 00000002

    pair_key = "_".join([date_part, loc1, loc2, frame_id])
    return pair_key


def find_matching_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng cặp IR/VIS (hoặc thermal/rgb) dựa trên pair_key.

    Không giới hạn theo source_name, vì IR & VIS có thể ở 2 prefix / source khác nhau.
    """
    df = df.copy()
    df["pair_key"] = df["filename"].apply(build_pair_key_from_filename)

    pivot = df.pivot_table(
        index="pair_key",
        columns="modality",
        values="id",
        aggfunc="first",
    )

    # chỉ giữ những pair có từ 2 modality trở lên
    pivot = pivot[pivot.count(axis=1) >= 2].reset_index()
    return pivot


def find_counterpart_image(df_all: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    """
    Tìm ảnh 'cùng scene khác modality' với row dựa trên pair_key.
    Không giới hạn theo source_name.
    """
    key = build_pair_key_from_filename(row["filename"])

    df = df_all.copy()
    df["pair_key"] = df["filename"].apply(build_pair_key_from_filename)

    candidates = df[df["pair_key"] == key]
    candidates = candidates[candidates["id"] != row["id"]]

    if candidates.empty:
        return None

    diff_mod = candidates[candidates["modality"] != row["modality"]]
    if not diff_mod.empty:
        return diff_mod.iloc[0]

    return candidates.iloc[0]

# =========================
# TAB 0 – System Architecture
# =========================

def page_system_architecture():
    st.header("System Architecture Diagram")

    st.markdown("""
    Đây là mô hình tổng quan của hệ thống Rescue Analytics:
    - Luồng dữ liệu từ S3 → ETL → PostgreSQL  
    - Các module analytics (PCA / t-SNE / feature stats)  
    - Dashboard IR/VIS switching  
    - Incremental S3 ETL  
    - Alert pipeline  
    """)

    try:
        with open("architecture/system_architecture.svg", "r") as f:
            svg = f.read()
        st.markdown(svg, unsafe_allow_html=True)
    except Exception as e:
        st.error("Không tìm thấy file system_architecture.svg. Hãy upload nó vào repo /architecture/")
        st.exception(e)


# =========================
# TAB 1 – Mission Overview
# =========================

def page_mission_overview():
    st.header("Mission Overview & Hotspots")

    df = load_images(limit=None)
    if df.empty:
        st.warning("No data in images table yet.")
        return

    total_images = len(df)
    total_with_person = int(df["has_person"].sum())
    total_boxes = load_total_person_boxes()
    persons_by_source = load_person_counts_by_source()
    num_hot_sources = int((persons_by_source["total_persons"] > 0).sum())

    st.subheader("Key Mission Stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total images", total_images)
    c2.metric("Images with person", total_with_person)
    c3.metric("Total persons (YOLO boxes)", total_boxes)
    c4.metric("Hotspot sources (≥1 person)", num_hot_sources)

    st.markdown("---")

    # coverage scatter
    st.subheader("Search Coverage (pseudo-map)")

    df_cov = df.copy()
    df_cov["index"] = range(1, len(df_cov) + 1)

    fig_cov = px.scatter(
        df_cov,
        x="index",
        y="source_name",
        color="has_person",
        size=df_cov["num_persons"].clip(lower=1),
        size_max=18,
        hover_data=["id", "filename", "modality", "num_persons"],
        title="Coverage along sequence (bigger marker = more persons)",
    )
    fig_cov.update_layout(height=350)
    st.plotly_chart(fig_cov, use_container_width=True)

    # heatmap persons per source x index
    st.markdown("### Hotspot heatmap (thermal style)")
    df_hm = df_cov.copy()
    heat = df_hm.pivot_table(
        index="source_name",
        columns="index",
        values="num_persons",
        aggfunc="sum",
    ).fillna(0)

    fig_hm = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Inferno",
        labels={"color": "# persons"},
        title="Heatmap of persons per image index and source",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    st.subheader("Hotspots by Source")

    col_left, col_right = st.columns(2)
    with col_left:
        if not persons_by_source.empty:
            fig_src = px.bar(
                persons_by_source,
                x="source_name",
                y="total_persons",
                title="Total persons per source",
            )
            fig_src.update_layout(xaxis_title="Source", yaxis_title="# persons")
            st.plotly_chart(fig_src, use_container_width=True)
        else:
            st.info("No per-source stats available.")

    with col_right:
        fig_np = px.histogram(
            df,
            x="num_persons",
            color="modality",
            title="Distribution of persons per image",
        )
        fig_np.update_layout(xaxis_title="num_persons", yaxis_title="count")
        st.plotly_chart(fig_np, use_container_width=True)


# =========================
# TAB 2 – Sensor Analytics
# =========================

def page_sensor_analytics():
    st.header("Sensor & Environment Analytics (IR vs VIS)")

    df = load_images(limit=None)
    if df.empty:
        st.warning("No data in images table yet.")
        return

    st.subheader("Side-by-side scene comparison")

    pairs = find_matching_pairs(df)

    if pairs.empty:
        st.info("No multi-modality pairs detected (based on last 8-digit id). Showing single image.")
        ids = df["id"].tolist()
        image_id = st.selectbox("Select image id", ids)
        row = df[df["id"] == image_id].iloc[0]
        img = load_image_from_s3(row["s3_key"])
        boxes = load_yolo_boxes_for_image(int(row["id"])).to_dict(orient="records")
        if boxes:
            fig = draw_boxes_on_image(img, boxes)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image(img, caption=row["filename"], use_column_width=True)
    else:
        pair_keys = pairs["pair_key"].unique().tolist()
        pk = st.selectbox("Choose a paired scene (last id key)", pair_keys)

        row_pair = pairs[pairs["pair_key"] == pk].iloc[0]

        # Các modality có mặt trong pair này (vd: "thermal", "rgb")
        mod_cols = [c for c in pairs.columns if c not in ["pair_key"]]

        # Mặc định: thermal bên trái, rgb/vís bên phải nếu có
        # Nếu tên modality khác, vẫn hiển thị bình thường
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Left view (first modality)**")
            left_mod = mod_cols[0]
            if not pd.isna(row_pair.get(left_mod)):
                left_id = int(row_pair[left_mod])
                row_left = df[df["id"] == left_id].iloc[0]
                img_left = load_image_from_s3(row_left["s3_key"])
                boxes_left = load_yolo_boxes_for_image(left_id).to_dict(orient="records")
                if boxes_left:
                    fig_left = draw_boxes_on_image(img_left, boxes_left)
                    st.plotly_chart(fig_left, use_container_width=True)
                else:
                    st.image(img_left, caption=f"{row_left['filename']} ({left_mod})", use_column_width=True)
            else:
                st.info("No image for this modality.")

        with col2:
            st.markdown("**Right view (second modality)**")
            if len(mod_cols) > 1:
                right_mod = mod_cols[1]
                if not pd.isna(row_pair.get(right_mod)):
                    right_id = int(row_pair[right_mod])
                    row_right = df[df["id"] == right_id].iloc[0]
                    img_right = load_image_from_s3(row_right["s3_key"])
                    boxes_right = load_yolo_boxes_for_image(right_id).to_dict(orient="records")
                    if boxes_right:
                        fig_right = draw_boxes_on_image(img_right, boxes_right)
                        st.plotly_chart(fig_right, use_container_width=True)
                    else:
                        st.image(
                            img_right,
                            caption=f"{row_right['filename']} ({right_mod})",
                            use_column_width=True,
                        )
                else:
                    st.info("No second modality image for this pair.")
            else:
                st.info("Only one modality in this pair.")

    st.markdown("---")
    st.subheader("Modality vs environment statistics")

    fig_bright = px.histogram(
        df,
        x="brightness",
        color="modality",
        nbins=40,
        title="Brightness distribution by modality",
    )
    st.plotly_chart(fig_bright, use_container_width=True)

    fig_person = px.histogram(
        df,
        x="num_persons",
        color="modality",
        barmode="group",
        title="Persons per image by modality",
    )
    st.plotly_chart(fig_person, use_container_width=True)

    st.markdown("### Auto insights")
    by_mod = (
        df.groupby("modality")
        .agg(
            mean_brightness=("brightness", "mean"),
            pct_with_person=("has_person", "mean"),
        )
        .reset_index()
    )
    st.dataframe(by_mod, use_container_width=True)


# =========================
# TAB 3 – Scene Embeddings
# =========================

def page_scene_embeddings():
    st.header("Scene Embeddings (PCA / t-SNE)")

    method = st.radio("Embedding method", ["PCA", "t-SNE"], horizontal=True)

    filter_hp = st.selectbox(
        "Filter by has_person",
        ["All", "Only has person", "Only no person"],
    )
    only_has_person = None
    if filter_hp == "Only has person":
        only_has_person = True
    elif filter_hp == "Only no person":
        only_has_person = False

    max_points = st.slider(
        "Max points to use (too many may be slow)",
        min_value=200,
        max_value=5000,
        value=1500,
        step=100,
    )

    if st.button("Compute embedding"):
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
                    title="PCA 2D embedding of scenes",
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
                    title="t-SNE 2D embedding of scenes",
                )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature statistics (for feature selection discussion)")
        stats_df = simple_feature_stats()
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("### Drill-down: inspect one embedded scene")
        image_ids = df_emb["id"].tolist()
        chosen_id = st.selectbox("Choose image id", image_ids)
        row = df_emb[df_emb["id"] == chosen_id].iloc[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            img = load_image_from_s3(row["s3_key"])
            boxes_df = load_yolo_boxes_for_image(int(row["id"]))
            boxes = boxes_df.to_dict(orient="records")
            if boxes:
                fig_img = draw_boxes_on_image(img, boxes)
                st.plotly_chart(fig_img, use_container_width=True)
            else:
                st.image(img, caption=row["filename"], use_column_width=True)

        with col2:
            st.write(
                {
                    "id": int(row["id"]),
                    "source_name": row["source_name"],
                    "filename": row["filename"],
                    "modality": row["modality"],
                    "has_person": bool(row["has_person"]),
                    "num_persons": int(row["num_persons"]),
                    "brightness": float(row["brightness"]),
                }
            )


# =========================
# TAB 4 – Image & Rescue Explorer
# =========================

def page_image_explorer():
    st.header("Image & Rescue Explorer")

    df_all = load_images(limit=None)
    if df_all.empty:
        st.warning("No data in images table yet.")
        return

    # Sidebar filters
    st.sidebar.subheader("Filters")

    modalities = ["All"] + sorted(df_all["modality"].dropna().unique().tolist())
    selected_mod = st.sidebar.selectbox("Modality", modalities)

    sources = ["All"] + sorted(df_all["source_name"].dropna().unique().tolist())
    selected_src = st.sidebar.selectbox("Source", sources)

    has_person_opt = st.sidebar.selectbox(
        "Rescue cases (proxy: has_person)",
        ["All", "Only rescue (has_person)", "Only non-rescue"],
    )

    filtered = df_all.copy()
    if selected_mod != "All":
        filtered = filtered[filtered["modality"] == selected_mod]
    if selected_src != "All":
        filtered = filtered[filtered["source_name"] == selected_src]
    if has_person_opt == "Only rescue (has_person)":
        filtered = filtered[filtered["has_person"] == True]
    elif has_person_opt == "Only non-rescue":
        filtered = filtered[filtered["has_person"] == False]

    st.write(f"{len(filtered)} images match filters.")

    if filtered.empty:
        return

    image_ids = filtered["id"].tolist()
    selected_id = st.selectbox("Select image id", image_ids)
    base_row = filtered[filtered["id"] == selected_id].iloc[0]

    # tìm counterpart IR/VIS nếu có
    counterpart = find_counterpart_image(df_all, base_row)

    # list options cho toggle sensor
    view_options = [(base_row["modality"], base_row)]
    if counterpart is not None:
        view_options.append((counterpart["modality"], counterpart))

    labels = [m for m, _ in view_options]
    default_idx = 0

    st.write("### Sensor view")
    chosen_label = st.radio(
        "Modality",
        labels,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    row_to_show = next(r for m, r in view_options if m == chosen_label)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"Image ({row_to_show['modality']}) with bounding boxes")
        img = load_image_from_s3(row_to_show["s3_key"])
        boxes_df = load_yolo_boxes_for_image(int(row_to_show["id"]))
        boxes = boxes_df.to_dict(orient="records")
        if boxes:
            fig = draw_boxes_on_image(img, boxes)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image(img, caption=row_to_show["filename"], use_column_width=True)

    with col_right:
        st.subheader("Metadata")
        st.write(
            {
                "id": int(row_to_show["id"]),
                "source_name": row_to_show["source_name"],
                "filename": row_to_show["filename"],
                "modality": row_to_show["modality"],
                "has_person": bool(row_to_show["has_person"]),
                "num_persons": int(row_to_show["num_persons"]),
                "brightness": float(row_to_show["brightness"]),
                "width": int(row_to_show["width"]),
                "height": int(row_to_show["height"]),
            }
        )

        st.subheader("YOLO boxes")
        if boxes_df.empty:
            st.write("No YOLO boxes for this image.")
        else:
            st.dataframe(boxes_df)

        if row_to_show["has_person"]:
            st.markdown("### ✅ RESCUE CASE (has_person = True)")
        else:
            st.markdown("### ⚪ Non-rescue image")


# =========================
# Main
# =========================

def main():
    st.set_page_config(
        page_title="Rescue Visual Analytics Console",
        layout="wide",
    )

    st.sidebar.write("DB host in app:", settings.db.host)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["System Architecture", "Mission Overview", "Sensor Analytics", "Scene Embeddings", "Image Explorer"],
    )

    if page == "Mission Overview":
        page_mission_overview()
    elif page == "Sensor Analytics":
        page_sensor_analytics()
    elif page == "Scene Embeddings":
        page_scene_embeddings()
    elif page == "Image Explorer":
        page_image_explorer()
    elif page == "System Architecture":
        page_system_architecture()


if __name__ == "__main__":
    main()
