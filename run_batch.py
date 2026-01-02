import sys
import polars as pl
from pathlib import Path
from core.engine import AttendanceEngine

def main():
    engine = AttendanceEngine()
    video_dir = Path("videos")
    csv_path = Path("Attendance_Report.csv")
    
    # 1. 确定待处理视频
    if len(sys.argv) > 1:
        target_videos = [video_dir / arg for arg in sys.argv[1:] if (video_dir / arg).exists()]
    else:
        target_videos = sorted(list(video_dir.glob("*.mp4")))

    if not target_videos:
        print("No valid videos to process.")
        return

    # 2. 基础数据准备 (始终以最新的 DB 为准)
    base_info = engine.db.select(["id", "name"])

    # 3. 加载旧报表并合并
    if csv_path.exists():
        old_report = pl.read_csv(csv_path)
        # 获取 CSV 中已有的视频列（排除掉统计列和基础信息列）
        video_cols_exist = [c for c in old_report.columns if c.endswith(".mp4")]
        # 将旧视频数据合并到最新的学生名单上
        report = base_info.join(
            old_report.select(["id"] + video_cols_exist), on="id", how="left"
        ).fill_null(0)
    else:
        report = base_info

    # 4. 处理当前传入的视频
    for v_path in target_videos:
        v_name = v_path.name
        print(f"[*] Processing: {v_name}")
        
        results = engine.process_video(v_path)
        v_col_values = [results[sid] for sid in engine.ids]
        
        # 覆盖逻辑
        if v_name in report.columns:
            report = report.drop(v_name)
        
        report = report.with_columns([pl.Series(name=v_name, values=v_col_values)])

    # 5. 动态计算统计列
    video_cols = sorted([c for c in report.columns if c.endswith(".mp4")])
    num_videos = len(video_cols)

    if video_cols:
        # Total_Hits: 所有视频的命中总和
        report = report.with_columns([
            pl.sum_horizontal(video_cols).alias("Total_Hits")
        ])
        
        # Attendance_Count: 在多少个视频里出现过 (命中数 > 0)
        report = report.with_columns([
            pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in video_cols]).alias("Attendance_Count")
        ])
        
        # Absence_Count: 未出席的视频数
        report = report.with_columns([
            (num_videos - pl.col("Attendance_Count")).alias("Absence_Count")
        ])

    # 6. 重新排列列顺序
    final_cols = ["id", "name"] + video_cols + ["Total_Hits", "Attendance_Count", "Absence_Count"]
    report = report.select(final_cols).sort("id")

    # 7. 保存并打印 (移除不兼容的 expand_columns 参数)
    report.write_csv(csv_path)
    
    print("\n" + "="*80)
    print(f"📊 考勤汇总报告 (共包含 {num_videos} 个视频源)")
    print("="*80)
    
    # 使用最基础的配置参数
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=50):
        print(report)
    
    print("="*80)
    print(f"Update complete. File: {csv_path}")

if __name__ == "__main__":
    main()
