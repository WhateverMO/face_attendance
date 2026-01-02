from insightface.app import FaceAnalysis
from .config import Config


class FaceProcessor:
    def __init__(self):
        self.app = FaceAnalysis(name=Config.NAME, providers=Config.get_providers())
        self.app.prepare(ctx_id=0, det_size=Config.DET_SIZE)

    def get_faces(self, frame):
        """获取一帧图像中的所有人脸及其特征"""
        return self.app.get(frame)

    @staticmethod
    def get_best_face(faces):
        """从多张脸中选出最大的一张"""
        if not faces:
            return None
        return sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )[0]
