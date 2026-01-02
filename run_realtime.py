import os
import sys
import cv2
import time
import polars as pl
from pathlib import Path
from core.engine import AttendanceEngine

def main():
    # 1. 初始化引擎
    try:
        engine = AttendanceEngine()
    except Exception as e:
        print(f"错误: {e}")
        return

    # 2. 初始化实时计数器
    # 使用字典存储 {id: hits}
    realtime_hits = {sid: 0 for sid in engine.ids}
    
    # 3. 打开摄像头
    cap = cv2.VideoCapture(0) # 0 为默认摄像头
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    print("\n" + "="*50)
    print("🚀 实时人脸签到系统已启动")
    print("操作提示: [S] 保存当前报表 | [Q] 退出系统")
    print("="*50 + "\n")

    last_ui_update = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 实时识别逻辑
            faces = engine.processor.get_faces(frame)
            for face in faces:
                stu_id, score = engine.identify_face(face.normed_embedding)
                if stu_id:
                    realtime_hits[stu_id] += 1
            
            # 在视频窗口显示（可选，方便对齐）
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Real-time Attendance (Press Q to Quit)', frame)

            # 每隔 1 秒在控制台刷新一次完整报表
            if time.time() - last_ui_update > 1.0:
                # 构造当前数据的 DataFrame
                data = []
                for sid, name in zip(engine.ids, engine.names):
                    hits = realtime_hits[sid]
                    status = "✅ Present" if hits >= 2 else "❌ Absent"
                    data.append({"ID": sid, "Name": name, "Hits": hits, "Status": status})
                
                df = pl.DataFrame(data).sort("Hits", descending=True)
                
                # 清屏指令（根据系统选择）
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"--- 实时考勤报表 ({time.strftime('%H:%M:%S')}) ---")
                with pl.Config(tbl_rows=-1, fmt_str_lengths=100):
                    print(df)
                print("\n[S] 保存 | [Q] 退出")
                last_ui_update = time.time()

            # 按键逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"Realtime_Report_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                df.write_csv(save_path)
                print(f"\n[!] 报表已保存至: {save_path}")
                time.sleep(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n系统已安全关闭。")

if __name__ == "__main__":
    main()
