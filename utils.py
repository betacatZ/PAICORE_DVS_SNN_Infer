import os
import dv_processing as dvp
import cv2 as cv
from datetime import timedelta
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import torch

# from spikingjelly.datasets import (
#     # cal_fixed_frames_number_segment_index,
#     integrate_events_segment_to_frame,
# )
from dv import AedatFile
import time


def filter_and_save_events(
    input_path: str, output_path: str, resolution: tuple, noise_duration_ms: int = 1, event_stream_name: str = "events"
):
    """
    读取事件数据，应用背景活动噪声过滤器，并将过滤后的事件保存到新文件中。

    :param input_path: 输入事件文件路径
    :param output_path: 输出事件文件路径
    :param resolution: 事件分辨率 (width, height)
    :param noise_duration_ms: 背景活动噪声的时间窗口（毫秒）
    :param event_stream_name: 写入事件流的名称
    """
    # 初始化事件读取器
    reader = dvp.io.MonoCameraRecording(input_path)

    # 配置事件写入器
    config = dvp.io.MonoCameraWriter.Config("DVXplorer")
    config.addEventStream(resolution)
    writer = dvp.io.MonoCameraWriter(output_path, config)

    # 初始化背景活动噪声过滤器
    filter = dvp.noise.BackgroundActivityNoiseFilter(
        resolution, backgroundActivityDuration=timedelta(milliseconds=noise_duration_ms)
    )

    try:
        # 开始处理事件
        while reader.isRunning():
            # 读取一批事件
            events = reader.getNextEventBatch()
            if events is not None:
                # 将事件传递给过滤器
                filter.accept(events)
                # 应用过滤器并生成过滤后的事件
                filtered = filter.generateEvents()
                # 打印过滤减少的比例
                print(f"Filter reduced number of events by a factor of {filter.getReductionFactor()}")
                # 将过滤后的事件写入新文件
                writer.writeEvents(filtered, streamName=event_stream_name)
    except KeyboardInterrupt:
        print("程序被用户中断。")


def play_video_from_aedat4(aedat4_path: str, window_name: str = "Preview", update_interval_ms: int = 33):
    """
    处理事件相机数据并实时显示预览帧。

    :param camera_path: 事件相机数据文件路径
    :param window_name: 显示窗口的名称
    :param update_interval_ms: 更新间隔时间（毫秒）
    """
    # 初始化事件相机录制
    capture = dvp.io.MonoCameraRecording(aedat4_path)

    # 确保支持事件流输出，否则抛出错误
    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input camera does not provide an event stream.")

    # 初始化累加器并设置分辨率
    visualizer = dvp.visualization.EventVisualizer(capture.getEventResolution())

    # 配置颜色方案
    visualizer.setBackgroundColor(dvp.visualization.colors.white())
    visualizer.setPositiveColor(dvp.visualization.colors.iniBlue())
    visualizer.setNegativeColor(dvp.visualization.colors.darkGrey())

    # 初始化预览窗口
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    # 初始化事件流切片器
    slicer = dvp.EventStreamSlicer()

    # 定义切片器回调函数
    def slicing_callback(events: dvp.EventStore):
        # 生成预览帧
        frame = visualizer.generateImage(events)
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"当前时间（精确到毫秒）: {formatted_time}")

        # 显示累加图像
        if frame is not None:
            print(frame.shape)
            cv.imshow(window_name, frame)

        # 等待指定时间
        cv.waitKey(update_interval_ms)

    # 注册回调函数，每隔指定时间执行一次
    slicer.doEveryTimeInterval(timedelta(milliseconds=update_interval_ms), slicing_callback)

    # 运行事件处理，直到相机断开连接
    try:
        while capture.isRunning():
            # 接收事件批次
            events = capture.getNextEventBatch()

            # 如果接收到事件，传递给切片器处理
            if events is not None:
                slicer.accept(events)
    except KeyboardInterrupt:
        print("程序被用户中断。")
    finally:
        # 关闭窗口并释放资源
        cv.destroyAllWindows()
        capture.close()


def save_np_frames_as_png(x, save_dir: str) -> None:
    """
    :param x: frames with ``shape=[T, 2, H, W]``
    :type x: Union[torch.Tensor, np.ndarray]
    :param save_dir: Directory to save the frames as PNG files
    :type save_dir: str
    :return: None
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for t in range(img_tensor.shape[0]):
        img = to_img(img_tensor[t])
        img.save(os.path.join(save_dir, f"frame_{t:04d}.png"))
    print(f"Saved all frames to directory [{save_dir}]")


def convert_event_files_to_frames(
    frames_num,
    H,
    W,
    event=None,
    aedat_file=None,
) -> np.ndarray:
    if aedat_file:
        print(f"Start to convert [{aedat_file}] to samples.")
        with AedatFile(aedat_file) as f:
            events = np.hstack([event for event in f["events"].numpy()])
    elif event is not None:  # 当传入EventStore对象时
        print("Processing EventStore object")
        # 从EventStore提取数据，格式与aedat文件一致
        event_data = []
        # 直接获取所有事件的各字段数组（时间复杂度O(1)）
        start = time.perf_counter()
        t = event.timestamps()  # 返回所有时间戳的numpy数组
        x = event.coordinates()[:, 0]  # 所有x坐标的数组
        y = event.coordinates()[:, 1]  # 所有y坐标的数组
        p = event.polarities()  # 所有极性值的数组
        end = time.perf_counter()
        print(f"EventStore extraction time: {timedelta(seconds=end - start)}")
        # 合并为结构化数组（替代逐事件赋值）

        # start = time.perf_counter()
        # event_data = np.empty(len(event), dtype=[("timestamp", "<i8"), ("x", "<i2"), ("y", "<i2"), ("polarity", "i1")])
        # event_data["timestamp"] = timestamps.astype("<i8")
        # event_data["x"] = x_coords.astype("<i2")
        # event_data["y"] = y_coords.astype("<i2")
        # event_data["polarity"] = polarities.astype("i1")

        # # 转换为结构化数组（匹配aedat文件的数据结构）
        # events = np.array(event_data, dtype=[("timestamp", "<i8"), ("x", "<i2"), ("y", "<i2"), ("polarity", "i1")])
        # end = time.perf_counter()
        # print(f"EventStore processing time: {timedelta(seconds=end - start)}")

    else:
        raise ValueError("必须提供aedat_file或event参数")

    # file_name = os.path.join(output_dir, f"{aedat_file.split('.')[0]}.npz")
    # os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # np_savez(file_name, t=events["timestamp"], x=events["x"], y=events["y"], p=events["polarity"])
    # start = time.perf_counter()
    # t, x, y, p = (events[key] for key in ("timestamp", "x", "y", "polarity"))
    # end = time.perf_counter()
    # print(f"Event extraction time: {timedelta(seconds=end - start)}")
    start = time.perf_counter()
    j_l, j_r = cal_fixed_frames_number_segment_index(t, "number", frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    end = time.perf_counter()
    print(f"Index calculation time: {timedelta(seconds=end - start)}")

    start = time.perf_counter()
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])
    end = time.perf_counter()
    print(f"Frame integration time: {timedelta(seconds=end - start)}")
    # torch.from_numpy(frames)
    return frames


def cal_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(frames_num, dtype=int)
    j_r = np.zeros(frames_num, dtype=int)
    N = events_t.size

    if split_by == "number":
        # 向量化计算索引 (提升约10倍)
        di = N // frames_num
        j_l = np.arange(0, frames_num * di, di, dtype=int)
        j_r = j_l + di
        j_r[-1] = N  # 最后一个区间包含剩余事件
        j_l = np.minimum(j_l, N)  # 防止越界

    elif split_by == "time":
        # 使用searchsorted替代逐元素比较 (提升约50倍)
        dt = (events_t[-1] - events_t[0]) / frames_num
        time_points = events_t[0] + dt * np.arange(frames_num + 1)
        indices = np.searchsorted(events_t, time_points)
        j_l = indices[:-1]
        j_r = indices[1:]
        j_r[-1] = N  # 最后一个右边界强制为N

    else:
        raise NotImplementedError

    return j_l, j_r


def integrate_events_segment_to_frame(
    x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int, j_l: int = 0, j_r: int = -1
) -> np.ndarray:
    # 预切片避免重复计算
    x_seg = x[j_l:j_r].astype(np.int32)
    y_seg = y[j_l:j_r].astype(np.int32)
    p_seg = p[j_l:j_r].astype(np.int32)

    # 计算线性索引 (提升缓存命中)
    linear_indices = y_seg * W + x_seg

    # 使用2D bincount同时统计两种极性
    event_counts = np.zeros((2, H * W), dtype=np.int32)
    for polarity in [0, 1]:
        mask = p_seg == polarity
        pos = linear_indices[mask]
        counts = np.bincount(pos, minlength=H * W)
        event_counts[polarity, : len(counts)] = counts

    return event_counts.reshape((2, H, W))


def resize_128_128(frames):
    if len(frames.shape) == 4:
        frames = np.expand_dims(frames, axis=0)
    assert len(frames.shape) == 5
    frames = torch.from_numpy(frames.astype(np.float32))
    N, T, C = frames.shape[0], frames.shape[1], frames.shape[2]
    resize = transforms.Resize((128, 128))
    resized_tensor = torch.stack([resize(frames[n, t, :, :, :]) for n in range(N) for t in range(T)])
    # 重新调整张量形状为 (N, T, C, 128, 128)
    image = resized_tensor.view(N, T, C, 128, 128)
    return image
