import os
import sys
import numpy as np
import polars as pl
from pathlib import Path
from core.processor import FaceProcessor

class AttendanceEngine:
    def __init__(self, db_path="student_db.ipc", threshold=0.42):
        self.db_path = Path(db_path)
        self.threshold = threshold
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database {db_path} not found.")
        
        # 加载数据
        self.db = pl.read_ipc(self.db_path)
        self.db_embeddings = np.array(self.db["embedding"].to_list(), dtype=np.float32)
        self.ids = self.db["id"].to_list()
        self.names = self.db["name"].to_list()

        # 静默加载模型
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            self.processor = FaceProcessor()
            self.processor.app.prepare(ctx_id=0, det_size=(640, 640))
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

    def identify_face(self, face_embedding):
        sims = np.dot(self.db_embeddings, face_embedding)
        max_idx = np.argmax(sims)
        if sims[max_idx] > self.threshold:
            return self.ids[max_idx], sims[max_idx]
        return None, 0

    def update_student_feature(self, stu_id, new_embedding, momentum=0.05):
        """特征融合进化"""
        if stu_id in self.ids:
            idx = self.ids.index(stu_id)
            old_emb = self.db_embeddings[idx]
            updated_emb = (old_emb * (1 - momentum)) + (new_embedding * momentum)
            updated_emb = updated_emb / np.linalg.norm(updated_emb)
            self.db_embeddings[idx] = updated_emb

    def save_db(self):
        """安全保存：先写 .tmp 再覆盖"""
        updated_list = [emb.tolist() for emb in self.db_embeddings]
        self.db = self.db.with_columns([
            pl.Series(name="embedding", values=updated_list).cast(pl.List(pl.Float32))
        ])
        
        tmp_path = self.db_path.with_suffix(".tmp")
        self.db.write_ipc(tmp_path)
        if tmp_path.exists():
            os.replace(tmp_path, self.db_path)
