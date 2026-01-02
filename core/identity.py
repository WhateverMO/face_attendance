import os
import polars as pl
from pathlib import Path

class IdentityManager:
    @staticmethod
    def get_existing_ids(faces_dir="faces", temp_dir="temp_faces"):
        """扫描所有可能存在的 ID 目录，返回已占用的 ID 集合"""
        ids = set()
        for d_path in [Path(faces_dir), Path(temp_dir)]:
            if d_path.exists():
                for sub in d_path.iterdir():
                    id_file = sub / "id.txt"
                    if id_file.exists():
                        ids.add(id_file.read_text(encoding="utf-8").strip())
        return ids

    @staticmethod
    def generate_unique_id(forbidden_ids):
        """生成一个新的 STU_XXXX 格式的唯一 ID"""
        idx = 1
        while True:
            new_id = f"STU_{idx:04d}"
            if new_id not in forbidden_ids:
                return new_id
            idx += 1

    @staticmethod
    def save_id(path, student_id):
        """在指定目录保存 id.txt"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "id.txt").write_text(student_id, encoding="utf-8")

    @classmethod
    def create_empty_pool(cls, output_path, count, forbidden_ids):
        """预生成指定数量的空 ID 文件夹"""
        print(f"[INFO] Creating {count} empty student folders with unique IDs...")
        for i in range(count):
            # 这里的文件夹名使用 pool_ 前缀以区分聚类出来的文件夹
            pool_dir = Path(output_path) / f"pool_student_{i:02d}"
            new_id = cls.generate_unique_id(forbidden_ids)
            forbidden_ids.add(new_id)
            cls.save_id(pool_dir, new_id)
