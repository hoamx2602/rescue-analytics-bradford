# rescue_analytics/etl/extract_from_s3.py
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional, List

from PIL import Image
from sqlalchemy import text

from rescue_analytics.config import settings, DataSource
from rescue_analytics.db import engine
from rescue_analytics.s3_storage import get_s3_client


@dataclass
class ImageObject:
    source: DataSource
    image_key: str
    label_key: Optional[str]
    etag: str
    last_modified: str  # ISO string


def list_objects_for_source(source: DataSource) -> List[ImageObject]:
    """List tất cả cặp (image, txt) dưới prefix của 1 source."""
    s3 = get_s3_client()
    bucket = settings.s3.bucket
    prefix = source.prefix

    continuation_token = None
    tmp_index: Dict[str, Dict[str, Optional[str]]] = {}

    print(f"[{source.name}] Listing objects under prefix: {prefix}")

    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**params)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if "." not in fname:
                continue
            stem, ext = fname.rsplit(".", 1)
            ext = ext.lower()

            if stem not in tmp_index:
                tmp_index[stem] = {
                    "image_key": None,
                    "label_key": None,
                    "image_etag": None,
                    "label_etag": None,
                    "image_lm": None,
                    "label_lm": None,
                }

            if ext in ("jpg", "jpeg", "png"):
                tmp_index[stem]["image_key"] = key
                tmp_index[stem]["image_etag"] = obj["ETag"].strip('"')
                tmp_index[stem]["image_lm"] = obj["LastModified"].isoformat()
            elif ext == "txt":
                tmp_index[stem]["label_key"] = key
                tmp_index[stem]["label_etag"] = obj["ETag"].strip('"')
                tmp_index[stem]["label_lm"] = obj["LastModified"].isoformat()

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    result: List[ImageObject] = []

    for stem, info in tmp_index.items():
        if not info["image_key"]:
            continue

        # etag kết hợp: nếu ảnh hoặc txt đổi -> etag_combined đổi
        etags = [info["image_etag"], info["label_etag"]]
        etag_combined = "|".join(e for e in etags if e is not None)

        # last_modified = max(last_modified image, label)
        lms = [info["image_lm"], info["label_lm"]]
        last_modified = max(l for l in lms if l is not None)

        result.append(
            ImageObject(
                source=source,
                image_key=info["image_key"],
                label_key=info["label_key"],
                etag=etag_combined,
                last_modified=last_modified,
            )
        )

    print(f"[{source.name}] Found {len(result)} image objects")
    return result


def is_image_up_to_date(obj: ImageObject) -> bool:
    """Kiểm tra trong DB đã có record cùng s3_key + s3_etag chưa."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT s3_etag FROM images WHERE s3_key = :s3_key"),
            {"s3_key": obj.image_key},
        ).fetchone()

    if row is None:
        return False
    existing_etag = row[0]
    return existing_etag == obj.etag


def parse_yolo_labels(text_data: str) -> List[dict]:
    """Parse nội dung file .txt YOLO thành list box."""
    boxes = []
    for line in text_data.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:])
        boxes.append(
            {
                "class_id": class_id,
                "x_center": x_c,
                "y_center": y_c,
                "box_width": w,
                "box_height": h,
            }
        )
    return boxes


def process_image_object(obj: ImageObject):
    """Download từ S3, tính metadata, upsert vào DB."""
    s3 = get_s3_client()
    bucket = settings.s3.bucket

    # 1. Download ảnh
    img_resp = s3.get_object(Bucket=bucket, Key=obj.image_key)
    img_bytes = img_resp["Body"].read()
    with Image.open(BytesIO(img_bytes)) as im:
        width, height = im.size
        gray = im.convert("L")
        brightness = float(sum(gray.getdata())) / (width * height * 255.0)

    # 2. Download label (YOLO txt) nếu có
    has_person = False
    num_persons = 0
    boxes: List[dict] = []

    if obj.label_key:
        lbl_resp = s3.get_object(Bucket=bucket, Key=obj.label_key)
        txt = lbl_resp["Body"].read().decode("utf-8")
        boxes = parse_yolo_labels(txt)
        num_persons = len(boxes)
        has_person = num_persons > 0

    filename = obj.image_key.split("/")[-1]

    # 3. Upsert vào DB
    with engine.begin() as conn:
        # images
        result = conn.execute(
            text(
                """
                INSERT INTO images (
                    s3_key, source_name, filename, modality,
                    has_person, num_persons,
                    brightness, width, height,
                    s3_etag, s3_last_modified, created_at, updated_at
                )
                VALUES (
                    :s3_key, :source_name, :filename, :modality,
                    :has_person, :num_persons,
                    :brightness, :width, :height,
                    :s3_etag, :s3_last_modified, NOW(), NOW()
                )
                ON CONFLICT (s3_key) DO UPDATE SET
                    source_name = EXCLUDED.source_name,
                    filename = EXCLUDED.filename,
                    modality = EXCLUDED.modality,
                    has_person = EXCLUDED.has_person,
                    num_persons = EXCLUDED.num_persons,
                    brightness = EXCLUDED.brightness,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    s3_etag = EXCLUDED.s3_etag,
                    s3_last_modified = EXCLUDED.s3_last_modified,
                    updated_at = NOW()
                RETURNING id
                """
            ),
            {
                "s3_key": obj.image_key,
                "source_name": obj.source.name,
                "filename": filename,
                "modality": obj.source.modality,
                "has_person": has_person,
                "num_persons": num_persons,
                "brightness": brightness,
                "width": width,
                "height": height,
                "s3_etag": obj.etag,
                "s3_last_modified": obj.last_modified,
            },
        )
        image_id = result.fetchone()[0]

        # xóa yolo_boxes cũ (nếu reprocess) và insert lại
        conn.execute(
            text("DELETE FROM yolo_boxes WHERE image_id = :image_id"),
            {"image_id": image_id},
        )

        for box in boxes:
            conn.execute(
                text(
                    """
                    INSERT INTO yolo_boxes (
                        image_id, class_id,
                        x_center, y_center,
                        box_width, box_height
                    )
                    VALUES (
                        :image_id, :class_id,
                        :x_center, :y_center,
                        :box_width, :box_height
                    )
                    """
                ),
                {"image_id": image_id, **box},
            )

    print(
        f"Processed {obj.image_key} | persons={num_persons} | brightness={brightness:.3f}"
    )


def run_full_extract():
    """Chạy extract cho tất cả sources, chỉ xử lý object mới/changed."""
    total = 0
    skipped = 0

    for source in settings.sources:
        objs = list_objects_for_source(source)
        for obj in objs:
            if is_image_up_to_date(obj):
                skipped += 1
                continue
            process_image_object(obj)
            total += 1

    print(f"Extraction done. New/updated: {total}, skipped: {skipped}")
