import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.cluster import DBSCAN
from core.processor import FaceProcessor
from core.identity import IdentityManager
from core.utils import get_face_quality, correct_frame_rotation
from core.config import Config

# ================= 全局配置 =================
EMPTY_POOL_COUNT = 50      # 预生成的空学生文件夹数量
PROCESSED_VIDEOS = 2       # 处理前 N 个视频进行抓取
CLUSTER_EPS = 0.4          # 聚类阈值（越小越严格）
# ============================================

def run_auto_crop():
    processor = FaceProcessor()
    video_files = sorted(list(Path("videos").glob("*.mp4")))[:PROCESSED_VIDEOS]
    output_path = Path("temp_faces")

    # 1. 重置 temp_faces 目录
    if output_path.exists(): 
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # 2. 初始化已占用的 ID 池
    forbidden_ids = IdentityManager.get_existing_ids()

    # 3. 执行功能：预生成 50 个空 ID 文件夹
    IdentityManager.create_empty_pool(output_path, EMPTY_POOL_COUNT, forbidden_ids)

    # 4. 特征提取
    extracted_data = []
    for v_p in video_files:
        print(f"[EXTRACT] {v_p.name}")
        cap = cv2.VideoCapture(str(v_p))
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # 采样率：每秒抓取 5 帧
            if f_idx % max(1, int(fps / 5)) == 0:
                frame = correct_frame_rotation(frame, w, h)
                for face in processor.get_faces(frame):
                    q = get_face_quality(face, frame)
                    if q > 12: # 质量过滤
                        b = face.bbox.astype(int)
                        # 扩边裁剪
                        crop = frame[max(0, b[1]-40):b[3]+40, max(0, b[0]-40):b[2]+40].copy()
                        extracted_data.append({"emb": face.normed_embedding, "img": crop, "q": q})
            f_idx += 1
        cap.release()

    # 5. 聚类并为聚类结果分配 ID
    if extracted_data:
        print(f"[INFO] Clustering {len(extracted_data)} face captures...")
        embs = np.array([d["emb"] for d in extracted_data])
        labels = DBSCAN(eps=CLUSTER_EPS, min_samples=3, metric="cosine").fit(embs).labels_

        unique_labels = [l for l in set(labels) if l != -1]
        for label in unique_labels:
            # 为聚类结果生成新 ID
            new_id = IdentityManager.generate_unique_id(forbidden_ids)
            forbidden_ids.add(new_id)
            
            cluster_dir = output_path / f"cluster_student_{label:02d}"
            IdentityManager.save_id(cluster_dir, new_id)
            
            # 存入该聚类中质量最好的前 5 张照片
            idxs = [i for i, l in enumerate(labels) if l == label]
            idxs = sorted(idxs, key=lambda i: extracted_data[i]["q"], reverse=True)
            for rank, idx in enumerate(idxs[:5]):
                cv2.imwrite(str(cluster_dir / f"rank{rank}.jpg"), extracted_data[idx]["img"])
        
        print(f"[SUCCESS] Generated {len(unique_labels)} clusters and {EMPTY_POOL_COUNT} empty ID folders.")

if __name__ == "__main__":
    run_auto_crop()
