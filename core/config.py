import onnxruntime as ort


class Config:
    NAME = "buffalo_l"
    DET_SIZE = (640, 640)
    # 识别阈值，建议 0.42-0.45
    SIMILARITY_THRESHOLD = 0.42
    # 特征进化动量因子
    EVOLUTION_MOMENTUM = 0.05
    # 高质量人脸判断标准
    QUALITY_SCORE_THRES = 0.65

    @staticmethod
    def get_providers():
        available = ort.get_available_providers()
        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in priority if p in available]
