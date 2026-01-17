import os
import cv2
import time
import polars as pl
from core.engine import AttendanceEngine
from core.config import Config


def main():
    engine = AttendanceEngine()
    realtime_hits = {sid: 0 for sid in engine.ids}
    cap = cv2.VideoCapture(0)

    print("[INFO] Real-time system started. Press 'S' to save, 'Q' to quit.")
    last_ui = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for face in engine.processor.get_faces(frame):
            sid, score = engine.identify_face(face.normed_embedding)
            if sid:
                realtime_hits[sid] += 1
                if score > Config.QUALITY_SCORE_THRES:
                    engine.update_student_feature(sid, face.normed_embedding)

            # 绘制框
            b = face.bbox.astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)

        if time.time() - last_ui > 1.5:
            os.system("cls" if os.name == "nt" else "clear")
            data = [
                {"ID": s, "Name": n, "Hits": realtime_hits[s]}
                for s, n in zip(engine.ids, engine.names)
            ]
            df = pl.DataFrame(data).sort("Hits", descending=True)
            print(f"--- Real-time Report ({time.strftime('%H:%M:%S')}) ---")
            with pl.Config(tbl_rows=-1):
                print(df)
            last_ui = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            engine.save_db()
            df.write_csv(f"Realtime_Report_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            print("[INFO] Snapshot saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
