import os
import sys
import logging
import warnings
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.cluster import DBSCAN

# 环境配置与日志屏蔽
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.processor import FaceProcessor
from core.identity import IdentityManager
from core.utils import get_face_quality, correct_frame_rotation

# ================= 业务配置 =================
PROCESSED_VIDEOS_COUNT = 2
EMPTY_ID_COUNT = 50  # 默认预生成 50 个空 ID 文件夹
SAMPLING_RATE = 6
CLUSTER_EPS = 0.4
# ============================================


def run_auto_crop():
    # 1. 屏蔽初始化输出
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        processor = FaceProcessor()
        processor.app.prepare(ctx_id=0, det_size=(1280, 1280))
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    # 2. 准备路径
    video_path = Path("videos")
    video_files = sorted(list(video_path.glob("*.mp4")))[:PROCESSED_VIDEOS_COUNT]
    output_path = Path("temp_faces")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # 3. ID 唯一性准备
    # 注意：这里会同时检查 faces 和刚清空的 temp_faces
    forbidden_ids = IdentityManager.get_existing_ids(
        faces_dir="faces", temp_dir="temp_faces"
    )
    logger.info(f"Initialized with {len(forbidden_ids)} existing IDs from library.")

    # 4. 预生成空 ID 文件夹池
    IdentityManager.create_empty_pool(output_path, EMPTY_ID_COUNT, forbidden_ids)

    # 5. 特征提取循环
    extracted_data = []
    for v_path in video_files:
        logger.info(f"Extracting faces from: {v_path.name}")
        cap = cv2.VideoCapture(str(v_path))
        fps, w, h = (
            cap.get(cv2.CAP_PROP_FPS),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = correct_frame_rotation(frame, w, h)

            if frame_idx % max(1, int(fps / SAMPLING_RATE)) == 0:
                faces = processor.get_faces(frame)
                for face in faces:
                    if face.det_score < 0.5:
                        continue
                    q = get_face_quality(face, frame)
                    if q < 15:
                        continue

                    bbox = face.bbox.astype(int)
                    h_img, w_img = frame.shape[:2]
                    nx1, ny1, nx2, ny2 = (
                        max(0, bbox[0] - 40),
                        max(0, bbox[1] - 40),
                        min(w_img, bbox[2] + 40),
                        min(h_img, bbox[3] + 40),
                    )
                    extracted_data.append(
                        {
                            "embedding": face.normed_embedding,
                            "image": frame[ny1:ny2, nx1:nx2].copy(),
                            "quality": q,
                        }
                    )
            frame_idx += 1
        cap.release()

    # 6. 聚类与已提取人脸的 ID 分配
    if extracted_data:
        logger.info(f"Clustering {len(extracted_data)} candidates...")
        embeddings = np.array([d["embedding"] for d in extracted_data])
        clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=2, metric="cosine").fit(
            embeddings
        )
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:
                continue
            indices = sorted(
                np.where(labels == label)[0],
                key=lambda i: extracted_data[i]["quality"],
                reverse=True,
            )

            student_dir = output_path / f"student_{label:02d}"
            new_id = IdentityManager.generate_unique_id(forbidden_ids)
            forbidden_ids.add(new_id)

            IdentityManager.save_id(student_dir, new_id)
            for rank, idx in enumerate(indices[:5]):
                cv2.imwrite(
                    str(student_dir / f"rank{rank}.jpg"), extracted_data[idx]["image"]
                )
            logger.info(f"Saved cluster {student_dir.name} with ID: {new_id}")

    logger.info("Auto crop and ID pool generation completed.")


if __name__ == "__main__":
    run_auto_crop()
