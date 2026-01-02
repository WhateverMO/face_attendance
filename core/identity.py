import os
import sys
import logging
import cv2
import numpy as np
import polars as pl
from pathlib import Path
from core.processor import FaceProcessor
from core.utils import correct_frame_rotation

# 日志与环境配置
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================= 配置参数 =================
DB_PATH = "student_db.ipc"
VIDEO_DIR = "videos"
THRESHOLD = 0.45  # 识别阈值，建议 0.4-0.5
MIN_OCCURRENCE = 3  # 视频中至少出现几次才算签到成功
SAMPLING_RATE = 2  # 签到识别采样率（每秒处理 2 帧，提高速度）
# ============================================


class AttendanceSystem:
    def __init__(self):
        # 1. 加载底库
        if not Path(DB_PATH).exists():
            raise FileNotFoundError(
                f"Database {DB_PATH} not found. Run register.py first."
            )

        self.db = pl.read_ipc(DB_PATH)
        self.db_embeddings = np.array(self.db["embedding"].to_list())
        self.names = self.db["name"].to_list()
        self.ids = self.db["id"].to_list()

        # 2. 初始化视觉模型
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            self.processor = FaceProcessor()
            self.processor.app.prepare(
                ctx_id=0, det_size=(640, 640)
            )  # 识别用 640 速度极快
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        logger.info(f"Attendance System Ready. Loaded {len(self.names)} students.")

    def process_single_video(self, video_path: Path):
        """核心函数：分析单个视频文件"""
        logger.info(f"Analyzing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 用于记录该视频中每个 ID 出现的次数
        video_counts = {stu_id: 0 for stu_id in self.ids}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 抽样处理提高效率
            if frame_idx % max(1, int(fps / SAMPLING_RATE)) == 0:
                frame = correct_frame_rotation(frame, w, h)
                faces = self.processor.get_faces(frame)

                for face in faces:
                    # 矩阵运算：计算当前脸与库中所有人的相似度
                    feat = face.normed_embedding
                    sims = np.dot(self.db_embeddings, feat)

                    max_idx = np.argmax(sims)
                    if sims[max_idx] > THRESHOLD:
                        target_id = self.ids[max_idx]
                        video_counts[target_id] += 1

            frame_idx += 1
        cap.release()
        return video_counts

    def run_batch_attendance(self, video_list: list):
        """批量处理循环"""
        # 结果汇总表：行是学生，列是视频
        total_report = self.db.select(["id", "name"])

        for v_path in video_list:
            counts_dict = self.process_single_video(v_path)

            # 将该视频的结果转为 Polars 列并合并
            v_column = [counts_dict[stu_id] for stu_id in self.ids]
            total_report = total_report.with_columns(
                [pl.Series(name=v_path.name, values=v_column)]
            )

        # 计算最终状态：在任意视频中出现次数 > MIN_OCCURRENCE 即为出勤
        video_cols = [c for c in total_report.columns if c not in ["id", "name"]]
        total_report = total_report.with_columns(
            [
                pl.fold(
                    acc=pl.lit(0), f=lambda acc, x: acc + x, exprs=pl.col(video_cols)
                ).alias("Total_Hits")
            ]
        )

        total_report = total_report.with_columns(
            [
                pl.when(pl.col("Total_Hits") >= MIN_OCCURRENCE)
                .then(pl.lit("Present"))
                .otherwise(pl.lit("Absent"))
                .alias("Status")
            ]
        )

        return total_report


if __name__ == "__main__":
    system = AttendanceSystem()

    # 获取视频列表（按名称排序）
    video_dir = Path(VIDEO_DIR)
    all_videos = sorted(list(video_dir.glob("*.mp4")))

    if not all_videos:
        logger.error("No videos to process.")
    else:
        # 执行批量处理
        report_df = system.run_batch_attendance(all_videos)

        # 保存结果
        report_df.write_csv("Attendance_Report.csv")
        # 如果安装了 xlsxwriter 也可以保存为 excel
        # report_df.write_excel("Attendance_Report.xlsx")

        logger.info("-" * 50)
        logger.info("Success! Attendance report generated.")
        print(report_df.select(["name", "Total_Hits", "Status"]))
