import sys
import cv2
import polars as pl
from pathlib import Path
from core.engine import AttendanceEngine

def sync_names(engine):
    """根据 faces/id.txt 强制同步内存中的 Name"""
    faces_path = Path("faces")
    if not faces_path.exists(): return
    
    id_name_map = {}
    for s_dir in faces_path.iterdir():
        if s_dir.is_dir() and (s_dir / "id.txt").exists():
            stu_id = (s_dir / "id.txt").read_text(encoding="utf-8").strip()
            id_name_map[stu_id] = s_dir.name
            
    if id_name_map:
        # 更新数据库中的名字
        engine.db = engine.db.with_columns([
            pl.col("id").replace(id_name_map, default=pl.col("name")).alias("name")
        ])
        engine.names = engine.db["name"].to_list()

def main():
    engine = AttendanceEngine()
    sync_names(engine) # 1. 同步最新名字
    
    video_dir = Path("videos")
    csv_path = Path("Attendance_Report.csv")
    
    # 2. 视频列表
    target_videos = [video_dir / arg for arg in sys.argv[1:] if (video_dir / arg).exists()] if len(sys.argv) > 1 else sorted(list(video_dir.glob("*.mp4")))
    if not target_videos: return

    # 3. 报表初始化
    if csv_path.exists():
        old_report = pl.read_csv(csv_path)
        v_cols = [c for c in old_report.columns if c.endswith(".mp4")]
        report = engine.db.select(["id", "name"]).join(old_report.select(["id"] + v_cols), on="id", how="left").fill_null(0)
    else:
        report = engine.db.select(["id", "name"])

    # 4. 分析与进化
    for v_path in target_videos:
        print(f"[*] Processing: {v_path.name}")
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        hit_counts = {sid: 0 for sid in engine.ids}
        
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if f_idx % max(1, int(fps/2)) == 0:
                faces = engine.processor.get_faces(frame)
                for face in faces:
                    sid, score = engine.identify_face(face.normed_embedding)
                    if sid:
                        hit_counts[sid] += 1
                        if score > 0.65: # 高质量脸进化
                            engine.update_student_feature(sid, face.normed_embedding)
            f_idx += 1
        cap.release()

        v_name = v_path.name
        v_data = [hit_counts[sid] for sid in engine.ids]
        if v_name in report.columns: report = report.drop(v_name)
        report = report.with_columns([pl.Series(name=v_name, values=v_data)])

    # 5. 统计排序
    v_cols = sorted([c for c in report.columns if c.endswith(".mp4")])
    report = report.with_columns([
        pl.sum_horizontal(v_cols).alias("Total_Hits"),
        pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in v_cols]).alias("Attendance_Count")
    ])
    report = report.with_columns([(len(v_cols) - pl.col("Attendance_Count")).alias("Absence_Count")])
    
    # 6. 保存所有结果
    engine.save_db() # 进化特征写回 IPC
    report.select(["id", "name"] + v_cols + ["Total_Hits", "Attendance_Count", "Absence_Count"]).sort("id").write_csv(csv_path)
    
    print("\n" + "="*50)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(report.select(["id", "name", "Total_Hits", "Attendance_Count", "Absence_Count"]))
    print("="*50)

if __name__ == "__main__":
    main()
