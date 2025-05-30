import os
import dv_processing as dv
import time
import numpy as np
from datetime import timedelta
from spikingjelly.activation_based import functional
import torch
from utils import convert_event_files_to_frames, resize_128_128, getNetParam
from torchvision import transforms
from quantize.qat_model import PAIBOX_DVSNet_FC, QAT_DVSNet, FC_Head
import cv2
from threading import Thread
import argparse
import yaml
import paibox as pb
from paiboard import PAIBoard_SIM
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


config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument(
    "-c",
    "--config",
    default="args.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)  # imagenet.yml  cifar10.yml
parser = argparse.ArgumentParser(description="PyTorch args")
parser.add_argument(
    "-T",
    "--time-step",
    type=int,
    default=16,
    metavar="time",
    help="simulation time step of spiking neuron (default: 16)",
)
parser.add_argument(
    "--num-classes", type=int, default=9, metavar="N", help="number of label classes (Model default if None)"
)
parser.add_argument(
    "--checkpoint_path",
    default="E:\\test\\code\\dvs_paibox_sim\\QAT_DVSNet_best.pth",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--mapper",
    default=False,
    type=str,
    help="是否导出映射文件",
)

parser.add_argument("--channels", default=32, help="channels")


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


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


def infer(model, pb_model_fc, fc_head, param_dict, device, frames):
    # frames = convert_aedat_files_to_frames(aedat4_file, 16, 260, 346)
    start = time.perf_counter()
    # total_pb_out = np.array([])
    image = resize_128_128(frames)
    # sim = pb.Simulator(pb_model_fc)
    image: functional.Tensor = image.transpose(0, 1).to(device, non_blocking=True)
    with torch.no_grad():
        conv_feature, output = model(image)
        
        total_pb_out = pb_model_fc(conv_feature)
        total_pb_out = torch.tensor(total_pb_out).unsqueeze(1).to(torch.float32)
        # for t, feature in enumerate(conv_feature):
        #     feature = feature.squeeze()  # torch.Size([1024])
        #     feature = feature.numpy().astype(np.uint8)
        #     pb_model_fc.inp.input = feature
        #     sim.run(1 + (0 if t + 1 != param_dict["timestep"] else param_dict["delay"]))

        # total_pb_out = sim.data[pb_model_fc.probe1].astype(np.float32)
        # total_pb_out = total_pb_out[param_dict["delay"] :]
        # total_pb_out = torch.tensor(total_pb_out).unsqueeze(1)

        fc_out = fc_head.fc_block(total_pb_out)
        out = fc_head.head(fc_out.mean(0))
        # sim.reset()
        functional.reset_net(fc_head)
        functional.reset_net(model)
    end = time.perf_counter()
    elapsed_microseconds = (end - start) * 1e3
    print(f"推理耗时: {elapsed_microseconds:.3f} 毫秒")
    cls = idx_to_class[out.argmax().int()]
    print(f"cls:{cls}\n")
    # print("cls:", idx_to_class[output.argmax().int()])
    # save_np_frames_as_png(frames, "./cc1111")


def DVS_PAIBOX_infer_sim(camera_name, duration, model, pb_model_fc, fc_head, param_dict, time_step, device):
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
                filter = dv.noise.BackgroundActivityNoiseFilter(
                    (346, 260), backgroundActivityDuration=timedelta(milliseconds=1)
                )
                filter.accept(events)
                filtered = filter.generateEvents()
                buffer.add(filtered)

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

                    frames = convert_event_files_to_frames(time_step, 260, 346, window)

                    end = time.perf_counter()
                    elapsed_microseconds = (end - start) * 1e3
                    i += 1
                    print(f"转换耗时: {elapsed_microseconds:.3f} 毫秒 i: {i}")
                    # dynamic_display_last_frame(frames)

                    # infer_thread(model, frames, device)
                    infer(model, pb_model_fc, fc_head, param_dict, device, frames)

                    # 维护缓冲区容量
                    if (buffer.getHighestTime() - buffer.getLowestTime()) > 5_000_000:
                        buffer = buffer.sliceTime(buffer.getHighestTime() - 5_000_000, buffer.getHighestTime())

                    last_write_time = current_time
                    file_count += 1

        print(f"Completed. Files generated: {file_count}")

    except KeyboardInterrupt:
        print("\nRecording aborted by user")


def main():
    args, args_text = _parse_args()
    print(args)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    param_dict = getNetParam(checkpoint, args.time_step, 1)
    baseDir = "./mapper/"

    pb_model_fc = PAIBoard_SIM(baseDir, args.time_step, layer_num=1)
    pb_model_fc.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    pb_model_fc.config(oFrmNum=10000)

    model = QAT_DVSNet(w_bits=8, channels=args.channels, n_cls=args.num_classes)
    # pb_model_fc = PAIBOX_DVSNet_FC(
    #     inp_shape=256,
    #     param_dict=param_dict,
    #     w_bits=8,
    # )
    # if args.mapper:
    #     mapper = pb.Mapper()
    #     mapper.build(pb_model_fc)
    #     graph_info = mapper.compile(
    #         core_estimate_only=False, weight_bit_optimization=True, grouping_optim_target="both"
    #     )
    #     mapper.export(write_to_file=True, fp="./mapper/", format="bin", split_by_chip=False, use_hw_sim=True)

    fc_head = FC_Head.from_qat_checkpoint(args.checkpoint_path, n_cls=args.num_classes)
    model.load_state_dict(checkpoint["model"])
    functional.set_step_mode(model, "m")
    functional.set_step_mode(fc_head, "m")
    device = torch.device("cpu")
    model.to(device)
    # DAVIS346_00000595
    DVS_PAIBOX_infer_sim("DAVIS346_00000595", None, model, pb_model_fc, fc_head, param_dict, args.time_step, device)


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
