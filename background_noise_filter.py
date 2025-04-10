import os
from utils import filter_and_save_events

def process_directory(origin_path: str, target_path: str, resolution: tuple, noise_duration_ms: int = 1, event_stream_name: str = "events"):
    """
    处理指定目录及其子目录中的所有 .aedat4 文件，并将处理后的文件保存到目标目录中，保持相同的目录结构。

    :param origin_path: 原始数据目录路径
    :param target_path: 目标数据目录路径
    :param resolution: 事件分辨率 (width, height)
    :param noise_duration_ms: 背景活动噪声的时间窗口（毫秒）
    :param event_stream_name: 写入事件流的名称
    """
    for root, dirs, files in os.walk(origin_path):
        # 计算当前目录在目标路径中的相对路径
        relative_path = os.path.relpath(root, origin_path)
        target_dir = os.path.join(target_path, relative_path)

        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.endswith(".aedat4"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, file)
                print(f"Processing file: {input_path} -> {output_path}")
                filter_and_save_events(input_path, output_path, resolution, noise_duration_ms, event_stream_name)

# 使用示例
if __name__ == "__main__":
    origin_path = "E:\\学习\\研究生\\毕设\\data\\origin_PAF"
    target_path = "E:\\学习\\研究生\\毕设\\data\\PAF_filter"
    resolution = (346, 260)
    noise_duration_ms = 1
    process_directory(origin_path, target_path, resolution, noise_duration_ms)