# rescue_analytics/viz/draw_boxes.py
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from rescue_analytics.s3_storage import get_s3_client
from rescue_analytics.config import settings


def load_image_from_s3(s3_key: str) -> Image.Image:
    s3 = get_s3_client()
    resp = s3.get_object(Bucket=settings.s3.bucket, Key=s3_key)
    img_bytes = resp["Body"].read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img


def yolo_to_xyxy(box, img_w, img_h):
    """
    YOLO (normalized) -> pixel (xmin, ymin, xmax, ymax)
    KHÔNG lật trục y, vì px.imshow giữ origin ở góc trên.
    """
    x_c = box["x_center"] * img_w
    y_c = box["y_center"] * img_h
    bw = box["box_width"] * img_w
    bh = box["box_height"] * img_h

    xmin = x_c - bw / 2
    xmax = x_c + bw / 2
    ymin = y_c - bh / 2
    ymax = y_c + bh / 2

    return xmin, ymin, xmax, ymax


def draw_boxes_on_image(img: Image.Image, boxes: List[dict]):
    """
    Trả về figure Plotly có overlay bounding boxes đúng vị trí.
    """
    w, h = img.size
    img_np = np.array(img)

    # Hiển thị ảnh bằng px.imshow để có hệ trục chuẩn cho ảnh
    fig = px.imshow(img_np)
    # Ẩn ticks, giữ origin ở góc trên
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=800,
        height=int(800 * h / w),
    )

    # Vẽ box
    for b in boxes:
        xmin, ymin, xmax, ymax = yolo_to_xyxy(b, w, h)

        fig.add_shape(
            type="rect",
            x0=xmin,
            y0=ymin,
            x1=xmax,
            y1=ymax,
            line=dict(color="red", width=3),
        )

        # Text label ở trên box
        fig.add_trace(
            go.Scatter(
                x=[xmin],
                y=[ymin - 3],
                text=[f"person ({b['class_id']})"],
                mode="text",
                textfont=dict(color="red", size=14),
                showlegend=False,
            )
        )

    return fig
