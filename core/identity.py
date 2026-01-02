import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IdentityManager:
    @staticmethod
    def get_existing_ids(faces_dir: str = "faces", temp_dir: str = "temp_faces") -> set:
        """扫描底库和临时目录，提取所有已分配的 ID，防止重复"""
        existing_ids = set()
        # 扫描正式底库
        for path in [Path(faces_dir), Path(temp_dir)]:
            if path.exists():
                for id_file in path.rglob("id.txt"):
                    try:
                        content = id_file.read_text(encoding="utf-8").strip()
                        if content:
                            existing_ids.add(content)
                    except Exception as e:
                        logger.warning(f"Failed to read ID file {id_file}: {e}")
        return existing_ids

    @staticmethod
    def generate_unique_id(forbidden_ids: set) -> str:
        """生成全局唯一的 ID 字符串"""
        while True:
            new_id = f"STU_{uuid.uuid4().hex[:8].upper()}"
            if new_id not in forbidden_ids:
                return new_id

    @staticmethod
    def save_id(folder_path: Path, student_id: str):
        """将 ID 写入文件夹中的 id.txt"""
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "id.txt").write_text(student_id, encoding="utf-8")

    @classmethod
    def create_empty_pool(cls, base_path: Path, count: int, forbidden_ids: set):
        """预生成指定数量的空 ID 文件夹"""
        pool_path = base_path / "empty_ids"
        pool_path.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            student_dir = pool_path / f"new_student_{i:02d}"
            new_id = cls.generate_unique_id(forbidden_ids)
            forbidden_ids.add(new_id)
            cls.save_id(student_dir, new_id)

        logger.info(f"Created {count} empty ID folders in {pool_path}")
