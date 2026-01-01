import os
import cv2
import polars as pl
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path


def get_optimal_providers():
    import onnxruntime as ort

    available = ort.get_available_providers()
    priority = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in priority if p in available]


def build_face_db(image_dir="faces", output_db="student_db.ipc"):
    # 1. 初始化模型
    providers = get_optimal_providers()
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_data = []
    image_path = Path(image_dir)

    if not image_path.exists():
        print(f"错误: 找不到文件夹 {image_dir}")
        return

    print(f"开始扫描底库照片: {image_dir} ...")

    for img_file in image_path.glob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            name = img_file.stem
            img = cv2.imread(str(img_file))

            if img is None:
                continue

            # 2. 提取特征
            faces = app.get(img)

            if len(faces) == 0:
                print(f"⚠️  警告: {img_file.name} 未检测到人脸，跳过")
                continue

            # 如果有多张脸，取画面占比最大的那张
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            embedding = faces[0].normed_embedding  # 512维归一化向量

            face_data.append({"name": name, "embedding": embedding.astype(np.float32)})
            print(f"✅ 已成功提取特征: {name}")

    if not face_data:
        print("❌ 未成功提取任何特征，请检查 faces 文件夹。")
        return

    # 3. 使用 Polars 转换为 DataFrame 并保存
    # 注意：embedding 列存入的是 list[float32]
    df = pl.DataFrame(face_data)

    # 保存为二进制 IPC 格式，读取速度极快
    df.write_ipc(output_db)
    print(f"\n成功! 底库已保存至 {output_db}，共 {len(face_data)} 位学生。")


if __name__ == "__main__":
    build_face_db()
