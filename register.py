import os
import sys
import logging
import warnings
import cv2
import numpy as np
import polars as pl
from pathlib import Path

os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.processor import FaceProcessor


def force_get_face(processor, img):
    """多策略识别：原图 -> 翻转 -> 加边框"""
    # 1. 原图尝试
    faces = processor.get_faces(img)
    if faces:
        return faces

    # 2. 翻转尝试
    faces = processor.get_faces(cv2.flip(img, 1))
    if faces:
        return faces

    # 3. 加边框尝试 (解决裁切过紧)
    h, w = img.shape[:2]
    border = int(min(h, w) * 0.2)
    padded_img = cv2.copyMakeBorder(
        img, border, border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    faces = processor.get_faces(padded_img)
    return faces


def run_registration():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        processor = FaceProcessor()
        # 使用最高检测精度
        processor.app.prepare(ctx_id=0, det_size=(1280, 1280))
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    faces_path = Path("faces")
    data = []
    student_dirs = sorted([d for d in faces_path.iterdir() if d.is_dir()])

    logger.info(f"Starting Robust Registration for {len(student_dirs)} folders...")

    for s_dir in student_dirs:
        name = s_dir.name
        id_file = s_dir / "id.txt"
        if not id_file.exists():
            continue
        student_id = id_file.read_text(encoding="utf-8").strip()

        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(s_dir.glob(ext)))

        embeddings = []
        for img_p in image_files:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            # 使用强力识别函数
            faces = force_get_face(processor, img)

            if faces:
                # 选最大的脸
                best_face = max(
                    faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                )
                embeddings.append(best_face.normed_embedding)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            final_emb = mean_emb / np.linalg.norm(mean_emb)
            data.append(
                {"id": student_id, "name": name, "embedding": final_emb.tolist()}
            )
            logger.info(f"Registered: {name:<12} | Samples: {len(embeddings)}")
        else:
            logger.error(f"FAILED: {name:<12} | Even with padding, no face found.")

    if data:
        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("embedding").cast(pl.List(pl.Float32)))
        df.write_ipc("student_db.ipc")
        logger.info("-" * 50)
        logger.info(f"Database saved with {len(data)} records.")


if __name__ == "__main__":
    run_registration()
