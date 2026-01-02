import os
import sys
import cv2
import numpy as np
import polars as pl
from pathlib import Path
from core.processor import FaceProcessor
from core.utils import correct_frame_rotation, get_face_quality


class AttendanceEngine:
    def __init__(self, db_path="student_db.ipc", threshold=0.42, sampling_rate=2):
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        # 加载数据库
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database {db_path} not found.")

        self.db = pl.read_ipc(db_path)
        self.db_embeddings = np.array(self.db["embedding"].to_list())
        self.ids = self.db["id"].to_list()
        self.names = self.db["name"].to_list()

        # 静默初始化模型
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            self.processor = FaceProcessor()
            self.processor.app.prepare(ctx_id=0, det_size=(640, 640))
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

    def identify_face(self, face_embedding):
        """核心特征比对"""
        sims = np.dot(self.db_embeddings, face_embedding)
        max_idx = np.argmax(sims)
        if sims[max_idx] > self.threshold:
            return self.ids[max_idx], sims[max_idx]
        return None, 0

    def process_video(self, video_path: Path):
        """处理单个视频，返回命中统计"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        hit_counts = {sid: 0 for sid in self.ids}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(1, int(fps / self.sampling_rate)) == 0:
                frame = correct_frame_rotation(frame, w, h)
                faces = self.processor.get_faces(frame)
                for face in faces:
                    stu_id, _ = self.identify_face(face.normed_embedding)
                    if stu_id:
                        hit_counts[stu_id] += 1
            frame_idx += 1
        cap.release()
        return hit_counts
