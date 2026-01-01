import insightface
from insightface.app import FaceAnalysis
import time
import numpy as np
import cv2
import onnxruntime as ort


def get_optimal_providers():
    """
    根据当前硬件环境，自动排列算力提供商的优先级
    """
    available = ort.get_available_providers()
    print(f"系统可用的 Providers: {available}")

    # 优先级定义：CUDA (Nvidia) > CoreML (Mac) > CPU
    priority_order = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # 过滤出当前系统支持的，并按优先级排序
    matched_providers = [p for p in priority_order if p in available]
    return matched_providers


def check_acceleration():
    providers = get_optimal_providers()
    print(f"选定的执行优先级: {providers}")

    # 初始化模型
    # ctx_id=0 表示使用第一个 GPU，如果没有 GPU 会自动回退
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        # det_size 必须是 32 的倍数，(640, 640) 是通用标准
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # 创建一个随机图像进行压力测试 (1080P 模拟)
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print("正在进行性能预热...")
    for _ in range(3):
        app.get(test_img)

    # 正式测试
    print("正在测试 10 帧处理速度...")
    start_time = time.time()
    for i in range(10):
        _ = app.get(test_img)

    total_time = time.time() - start_time
    avg_time = total_time / 10

    print(f"\n" + "=" * 30)
    print(f"硬件加速报告")
    print(f"=" * 30)
    print(f"当前运行设备: {app.models['detection'].session.get_providers()[0]}")
    print(f"平均每帧耗时: {avg_time:.4f} 秒")
    print(f"实时处理能力: {1 / avg_time:.2f} FPS")

    # 性能评估建议
    if 1 / avg_time > 15:
        print("性能评价: 🚀 极佳 (支持实时高帧率识别)")
    elif 1 / avg_time > 5:
        print("性能评价: ✅ 良好 (建议配合跳帧策略)")
    else:
        print("性能评价: 🐢 一般 (强烈建议开启跳帧处理)")
    print("=" * 30)


if __name__ == "__main__":
    check_acceleration()
