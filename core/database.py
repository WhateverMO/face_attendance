import polars as pl
import numpy as np


class FaceDatabase:
    def __init__(self, db_path="student_db.ipc"):
        self.db_path = db_path
        self.df = None
        self.embeddings = None
        self.names = None

    def load(self):
        self.df = pl.read_ipc(self.db_path)
        # 将 Polars 列转换为 Numpy 矩阵，方便后续矩阵乘法
        self.embeddings = np.array(self.df["embedding"].to_list())
        self.names = self.df["name"].to_list()
        return self

    def save(self, data):
        df = pl.DataFrame(data)
        df.write_ipc(self.db_path)
