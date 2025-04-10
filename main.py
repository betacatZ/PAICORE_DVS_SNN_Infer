import os
import dv_processing as dv
import time
import numpy as np
from datetime import timedelta
from spikingjelly.activation_based import functional
import torch
from utils import convert_event_files_to_frames, resize_128_128
from torchvision import transforms
from quantize.qat_model import QAT_DVSNet
import cv2
from threading import Thread

# cv2.setUseOptimized(True)  # 启用IPP优化
# cv2.setNumThreads(4)  # 根据CPU核心数设置

idx_to_class = [
    "arm_crossing",
    "get_up",
    "jumping",
    "kicking",
    "sit_down",
    "throwing",
    "turning_around",
    "walking",
    "waving",
]


def infer_thread(model, frames, device):
    Thread(target=infer, args=(model, device, frames)).start()


def manage_folder(folder="./data", max_files=15):
    """智能文件夹管理（自动清理旧文件）"""
    try:
        # 确保文件夹存在
        os.makedirs(folder, exist_ok=True)

        # 获取所有文件列表（排除子目录）
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        # 当文件数超过阈值时执行清理
        if len(files) > max_files:
            print(f"检测到文件数 {len(files)} > {max_files}，开始清理...")

            # 安全删除所有文件
            for filename in files:
                file_path = os.path.join(folder, filename)
                try:
                    os.unlink(file_path)  # 兼容Linux/Windows
                    print(f"已删除：{file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")

            print(f"清理完成，当前文件数：{len(os.listdir(folder))}")

    except Exception as e:
        print(f"文件夹管理错误: {e}")


def dynamic_display_last_frame(frames, window_name="Event Frame"):
    """
    动态显示事件帧的最后一帧 (与save_np_frames_as_png保持一致的通道处理逻辑)

    参数说明：
    frames - 输入帧序列 [T, 2, H, W] 或单帧 [2, H, W]
    window_name - 显示窗口名称（默认'Event Frame'）
    """
    # 硬件加速初始化
    cv2.setUseOptimized(True)  # 启用IPP优化[5](@ref)
    cv2.setNumThreads(4)  # 设置OpenCV线程数
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)[-1]

    # 提取最后一帧并保持4D张量

    # 创建RGB容器（与保存函数相同的处理逻辑）
    # img_tensor = torch.zeros((frames.shape[0], 3, frames.shape[2], frames.shape[3]))
    # img_tensor[:, 1] = frames[:, 0]  # 绿色通道 ← 输入通道0
    # img_tensor[:, 2] = frames[:, 1]  # 红色通道 ← 输入通道1
    img_tensor = torch.zeros((3, frames.shape[1], frames.shape[2]))
    img_tensor[1] = frames[0]  # 绿色通道 ← 输入通道0
    img_tensor[2] = frames[1]  # 红色通道 ← 输入通道1

    # 转换到PIL图像（复用保存函数的转换逻辑）
    to_img = transforms.ToPILImage()
    pil_img = to_img(img_tensor)

    # 转换为OpenCV格式
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 自动缩放窗口（保持与原始分辨率比例）
    h, w = cv_img.shape[:2]
    scale = 800 / max(h, w)
    resized_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))

    # 带时间戳的显示（每秒刷新率）
    timestamp = f"Frame: {-1} | Time: {time.strftime('%H:%M:%S')} jumping"
    cv2.putText(resized_img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

    # 显示处理
    cv2.imshow(window_name, resized_img)
    cv2.waitKey(1)  # 非阻塞刷新


def infer(model, device, frames):
    # frames = convert_aedat_files_to_frames(aedat4_file, 16, 260, 346)
    start = time.perf_counter()
    image = resize_128_128(frames)
    image: functional.Tensor = image.transpose(0, 1).to(device, non_blocking=True)
    _, output = model(image)
    functional.reset_net(model)
    end = time.perf_counter()
    elapsed_microseconds = (end - start) * 1e3
    print(f"推理耗时: {elapsed_microseconds:.3f} 毫秒")
    print("cls:", idx_to_class[output.argmax().int()])
    # save_np_frames_as_png(frames, "./cc1111")


def DVS_PAIBOX_infer_sim(camera_name, duration, model, device):
    try:
        # 初始化相机连接
        camera = dv.io.CameraCapture(camera_name)
        if not camera.isEventStreamAvailable():
            raise RuntimeError("Event stream not available")

        # 初始化缓冲区
        buffer = dv.EventStore()
        start_time = time.monotonic()
        last_write_time = None
        file_count = 0
        i = 0
        # 主循环条件处理
        while duration is None or (time.monotonic() - start_time < duration):
            # 读取事件数据
            events = camera.getNextEventBatch()
            if events is not None:
                start = time.perf_counter()
                filter = dv.noise.BackgroundActivityNoiseFilter(
                    (346, 260), backgroundActivityDuration=timedelta(milliseconds=1)
                )
                filter.accept(events)
                filtered = filter.generateEvents()
                buffer.add(filtered)
                end = time.perf_counter()
                elapsed_microseconds = (end - start) * 1e3
                # print(f"事件处理耗时: {elapsed_microseconds:.3f} 毫秒")

            current_time = time.monotonic()
            elapsed = current_time - start_time

            # 前5秒不保存（仅当有限时长时生效）
            if duration is not None and elapsed < 5:
                continue

            # 初始化首次写入时间
            if last_write_time is None:
                last_write_time = current_time

            # 精确312.5ms间隔保存
            if (current_time - last_write_time) >= 0.3125:
                if buffer.size() > 0:
                    # 计算时间窗口（当前时间前5秒）
                    end_timestamp = buffer.getHighestTime()
                    start_timestamp = end_timestamp - 5_000_000  # 5秒的微秒数

                    # 执行时间切片
                    window = buffer.sliceTime(start_timestamp, end_timestamp)
                    
                    start = time.perf_counter()

                    frames = convert_event_files_to_frames(16, 260, 346, window)

                    end = time.perf_counter()
                    elapsed_microseconds = (end - start) * 1e3
                    i += 1
                    print(f"转换耗时: {elapsed_microseconds:.3f} 毫秒 i: {i}")
                    # dynamic_display_last_frame(frames)

                    # infer_thread(model, frames, device)
                    infer(model, device, frames)

                    # 维护缓冲区容量
                    if (buffer.getHighestTime() - buffer.getLowestTime()) > 5_000_000:
                        buffer = buffer.sliceTime(buffer.getHighestTime() - 5_000_000, buffer.getHighestTime())

                    last_write_time = current_time
                    file_count += 1

        print(f"Completed. Files generated: {file_count}")

    except KeyboardInterrupt:
        print("\nRecording aborted by user")


def main():
    model = QAT_DVSNet(w_bits=8, channels=32, n_cls=9)
    functional.set_step_mode(model, "m")
    device = torch.device("cuda")
    model.to(device)
    # DAVIS346_00000595
    DVS_PAIBOX_infer_sim("DAVIS346_00000595", None, model, device)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Event recorder with sliding window")
    # parser.add_argument("-c", "--camera_name", default="DAVIS346_00000595", help="Camera identifier")
    # parser.add_argument("-o", "--output", required=True, help="Output file path prefix")
    # parser.add_argument(
    #     "-d", "--duration", type=float, default=None, help="Recording duration in seconds (omit for infinite)"
    # )
    # args = parser.parse_args()

    # record_events(args.camera_name, args.output, args.duration)
    # cameras = dv.io.discoverDevices()

    # print(f"Device discovery: found {len(cameras)} devices.")
    # for camera_name in cameras:
    #     print(f"Detected device [{camera_name}]")
    main()
