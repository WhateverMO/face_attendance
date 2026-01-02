import onnxruntime as ort


class Config:
    NAME = "buffalo_l"
    DET_SIZE = (640, 640)
    SIMILARITY_THRESHOLD = 0.45  # 签到阈值

    @staticmethod
    def get_providers():
        available = ort.get_available_providers()
        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in priority if p in available]
