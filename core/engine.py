import os
import sys
import numpy as np
import polars as pl
from pathlib import Path
from core.processor import FaceProcessor
from core.config import Config


class AttendanceEngine:
    def __init__(self, db_path="student_db.ipc"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database {db_path} not found.")

        self.db = pl.read_ipc(self.db_path)
        self.db_embeddings = np.array(self.db["embedding"].to_list(), dtype=np.float32)
        self.ids = self.db["id"].to_list()
        self.names = self.db["name"].to_list()

        # 静默加载视觉模型
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            self.processor = FaceProcessor()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

    def sync_names(self):
        """强制从 faces/ 目录同步最新的 Name"""
        faces_path = Path("faces")
        if not faces_path.exists():
            return
        id_map = {}
        for d in faces_path.iterdir():
            if d.is_dir() and (d / "id.txt").exists():
                sid = (d / "id.txt").read_text(encoding="utf-8").strip()
                id_map[sid] = d.name
        if id_map:
            self.db = self.db.with_columns(
                [pl.col("id").replace(id_map, default=pl.col("name")).alias("name")]
            )
            self.names = self.db["name"].to_list()

    def identify_face(self, face_embedding):
        sims = np.dot(self.db_embeddings, face_embedding)
        max_idx = np.argmax(sims)
        if sims[max_idx] > Config.SIMILARITY_THRESHOLD:
            return self.ids[max_idx], sims[max_idx]
        return None, 0

    def update_student_feature(self, stu_id, new_embedding):
        if stu_id in self.ids:
            idx = self.ids.index(stu_id)
            old_emb = self.db_embeddings[idx]
            m = Config.EVOLUTION_MOMENTUM
            updated = (old_emb * (1 - m)) + (new_embedding * m)
            self.db_embeddings[idx] = updated / np.linalg.norm(updated)

    def save_db(self):
        updated_list = [emb.tolist() for emb in self.db_embeddings]
        self.db = self.db.with_columns(
            [pl.Series(name="embedding", values=updated_list).cast(pl.List(pl.Float32))]
        )
        tmp_path = self.db_path.with_suffix(".tmp")
        self.db.write_ipc(tmp_path)
        if tmp_path.exists():
            os.replace(tmp_path, self.db_path)
