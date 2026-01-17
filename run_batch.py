import sys
import cv2
import polars as pl
from pathlib import Path
from core.engine import AttendanceEngine
from core.config import Config


def main():
    engine = AttendanceEngine()
    engine.sync_names()  # 自动同步 faces/ 目录的名字

    video_dir, csv_path = Path("videos"), Path("Attendance_Report.csv")
    targets = (
        [video_dir / a for a in sys.argv[1:] if (video_dir / a).exists()]
        if len(sys.argv) > 1
        else sorted(list(video_dir.glob("*.mp4")))
    )

    if not targets:
        return

    # 加载旧报表逻辑
    if csv_path.exists():
        old = pl.read_csv(csv_path)
        v_cols = [c for c in old.columns if c.endswith(".mp4")]
        report = (
            engine.db.select(["id", "name"])
            .join(old.select(["id"] + v_cols), on="id", how="left")
            .fill_null(0)
        )
    else:
        report = engine.db.select(["id", "name"])

    for v_p in targets:
        print(f"[PROCESS] {v_p.name}")
        cap = cv2.VideoCapture(str(v_p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        hits = {sid: 0 for sid in engine.ids}

        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % max(1, int(fps / 2)) == 0:
                for face in engine.processor.get_faces(frame):
                    sid, score = engine.identify_face(face.normed_embedding)
                    if sid:
                        hits[sid] += 1
                        if score > Config.QUALITY_SCORE_THRES:
                            engine.update_student_feature(sid, face.normed_embedding)
            f_idx += 1
        cap.release()

        if v_p.name in report.columns:
            report = report.drop(v_p.name)
        report = report.with_columns(
            [pl.Series(name=v_p.name, values=[hits[s] for s in engine.ids])]
        )

    # 统计排序并保存
    v_cols = sorted([c for c in report.columns if c.endswith(".mp4")])
    report = report.with_columns(
        [
            pl.sum_horizontal(v_cols).alias("Total_Hits"),
            pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in v_cols]).alias(
                "Attendance_Count"
            ),
        ]
    ).with_columns([(len(v_cols) - pl.col("Attendance_Count")).alias("Absence_Count")])

    engine.save_db()
    final = report.select(
        ["id", "name"] + v_cols + ["Total_Hits", "Attendance_Count", "Absence_Count"]
    ).sort("id")
    # final.write_csv(csv_path)
    # 1. 以标准的 utf-8 打开文件 (满足 Polars)
    with open(csv_path, "w", encoding="utf-8") as f:
        # 2. 手动写入 BOM 字符 (满足 Excel)
        f.write("\ufeff")
        # 3. 让 Polars 接着写入 CSV 数据
        final.write_csv(f)

    print("\n" + "=" * 70)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(final)
    print("=" * 70)


if __name__ == "__main__":
    main()
