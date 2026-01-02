import cv2
import numpy as np


def get_face_quality(face, frame) -> float:
    """评估人脸质量: 置信度 * 清晰度 * 尺寸"""
    bbox = face.bbox.astype(int)
    h_f, w_f = frame.shape[:2]

    # 边缘完整性检查
    if bbox[0] <= 2 or bbox[1] <= 2 or bbox[2] >= w_f - 2 or bbox[3] >= h_f - 2:
        return 0.01

    x1, y1, x2, y2 = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(w_f, bbox[2]),
        min(h_f, bbox[3]),
    )
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return 0

    # 清晰度 (Laplacian)
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    # 尺寸权重
    face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1000
    return face.det_score * sharpness * face_size


def correct_frame_rotation(frame, cap_width, cap_height):
    """根据宽高比纠正旋转"""
    if cap_width > cap_height:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame
