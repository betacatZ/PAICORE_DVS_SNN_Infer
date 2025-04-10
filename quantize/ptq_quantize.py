# 下述代码实现将原权重文件的所有参数缩放至-128到127后取整，并打印出新网络各层LIF的Vthr值。原权重文件的参数以float32格式存储，新权重文件的参数以int8格式存储。

import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from model import DVSNet, MainNet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def multply_model(model, multiplier):
    for name, param in model.items():
        if isinstance(param, torch.Tensor):
            # model[name] = (param * multiplier).to(torch.int8)
            model[name] = (param * multiplier).to(torch.int8)
    return model


def manual_multply_model(model, layer_name_list: list, mult_list: list):
    for layer_name, mult in zip(layer_name_list, mult_list):
        if layer_name + ".weight" in model:
            model[layer_name + ".weight"] = (model[layer_name + ".weight"] * mult).to(torch.int8)
            print("Layer " + layer_name + " weight multiplied by " + str(mult) + " and converted to int8.")
        else:
            print("Layer " + layer_name + " does not exist or has no weight.")
        if layer_name + ".bias" in model:
            model[layer_name + ".bias"] = (model[layer_name + ".bias"] * mult).to(torch.int8)
            print("Layer " + layer_name + " bias multiplied by " + str(mult) + " and converted to int8.")
        else:
            print("Layer " + layer_name + " does not exist or has no bias.")
    return model


def maxium_multply_model(model, sj_vthr: float = 1.0):
    """
    每层参数分别缩放并取整数，缩放系数由int(127.0/该层参数最大值)，该缩放系数作为新网络各层LIF的Vthr参数
    在缩放之前，需要将原网络参数除以sj_vthr，以使缩放时的sj_vthr归一化到1.0
    """
    layer_name_list = list(model.keys())
    layer_list = []
    seen = set()
    for name in layer_name_list:
        layer_name = name.rsplit(".", 1)[0]
        if layer_name not in seen:
            layer_list.append(layer_name)
            seen.add(layer_name)

    print(layer_list)
    vthr_list = []
    for layer_name in layer_list:
        max_weight = 0
        max_bias = 0
        if layer_name + ".weight" in model:
            weight_array = model[layer_name + ".weight"].detach().cpu().numpy()
            max_weight = np.max(np.abs(weight_array))
        if layer_name + ".bias" in model:
            bias_array = model[layer_name + ".bias"].detach().cpu().numpy()
            max_bias = np.max(np.abs(bias_array))
        max_value = max(max_weight, max_bias)
        # print(max_weight, max_bias)
        normalized_max_value = max_value / sj_vthr  # 归一化的最大值
        print(
            "Layer "
            + layer_name
            + " max value is "
            + str(max_value)
            + ", normalized max value is "
            + str(normalized_max_value)
        )
        normalized_mult = int(127.0 / float(normalized_max_value))
        mult = float(normalized_mult) / sj_vthr
        print(
            "Layer "
            + layer_name
            + " multiply factor is "
            + str(mult)
            + ", normalized multiply factor is "
            + str(normalized_mult)
        )
        if layer_name + ".weight" in model:
            model[layer_name + ".weight"] = (model[layer_name + ".weight"] * mult).to(torch.int8)

        if layer_name + ".bias" in model:
            model[layer_name + ".bias"] = (model[layer_name + ".bias"] * mult).to(torch.int8)

        # normalized_mult就是转换为int8之后新网络的vthr值
        vthr_list.append(normalized_mult)
    print(vthr_list)
    return model, vthr_list


def sj_inference(net, data_dir):
    net.eval()
    test_set = DVS128Gesture(root=data_dir, train=False, data_type="frame", frames_number=16, split_by="number")
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=True, drop_last=False, num_workers=1, pin_memory=True
    )
    test_acc_sj = 0
    test_samples = 0
    with torch.no_grad():
        for i, (image_tensor, label_tensor) in enumerate(test_data_loader):
            label = label_tensor[0]
            test_samples += 1
            frame = (
                torch.tensor(np.where(image_tensor.detach().numpy() == 0, 0, 1).astype(np.int8)).float().transpose(0, 1)
            )
            _, out_fr = net(frame)
            out_fr = out_fr.mean(0)
            pred_sj = out_fr.argmax(1).item()
            functional.reset_net(net)
            test_acc_sj += pred_sj == label
        total_acc_sj = test_acc_sj / test_samples
        print(f"test_acc_sj ={total_acc_sj}")


def main():
    # 加载原始权重文件
    checkpoint_path = "./logs/T16_b16_adam_lr0.001_c16_amp_cupy/checkpoint_max_convbn_fuse.pth"
    # 保存新的权重文件
    multiplied_checkpoint_path = "./logs/T16_b16_adam_lr0.001_c16_amp_cupy/checkpoint_max_quantize.pth"
    data_dir = "/home/zdm/dataset/DVS128Gesture"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 将所有参数乘以MULT后取整
    multiplied_checkpoint_sj = checkpoint.copy()
    # layer_name_list = ['conv.0', 'conv.2', 'conv.4', 'conv.6', 'conv.8', 'conv.10', 'fc.2', 'fc.5', 'fc.8']
    multiplied_checkpoint_sj["net"], multiplied_checkpoint_sj["vthr"] = maxium_multply_model(
        checkpoint["net"], sj_vthr=1.0
    )

    torch.save(multiplied_checkpoint_sj, multiplied_checkpoint_path)
    print("所有参数乘并转int8完成，新权重文件保存为 " + multiplied_checkpoint_path)

    checkpoint = torch.load(multiplied_checkpoint_path, map_location="cpu")
    net = MainNet(channels=16, checkpoint=checkpoint["net"], fuse=True, fuse_first=True, vthr=checkpoint["vthr"])
    sj_inference(net, "/home/zdm/dataset/DVS128Gesture")


if __name__ == "__main__":
    # checkpoint_path = './logs/T16_b16_adam_lr0.001_c128_amp_cupy/checkpoint_latest.pth'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # print(l)
    main()

# 运行结果：
# Layer conv.0 max value is 0.906594, normalized max value is 0.9065939784049988
# Layer conv.0 multiply factor is 140.0, normalized multiply factor is 140
# Layer conv.2 max value is 0.90164787, normalized max value is 0.9016478657722473
# Layer conv.2 multiply factor is 140.0, normalized multiply factor is 140
# Layer conv.4 max value is 1.305011, normalized max value is 1.3050110340118408
# Layer conv.4 multiply factor is 97.0, normalized multiply factor is 97
# Layer conv.6 max value is 1.000928, normalized max value is 1.0009280443191528
# Layer conv.6 multiply factor is 126.0, normalized multiply factor is 126
# Layer conv.8 max value is 1.7552505, normalized max value is 1.7552504539489746
# Layer conv.8 multiply factor is 72.0, normalized multiply factor is 72
# Layer conv.10 max value is 0.6547526, normalized max value is 0.6547526121139526
# Layer conv.10 multiply factor is 193.0, normalized multiply factor is 193
# Layer fc.2 max value is 0.053099077, normalized max value is 0.05309907719492912
# Layer fc.2 multiply factor is 2391.0, normalized multiply factor is 2391
# Layer fc.5 max value is 0.082026765, normalized max value is 0.08202676475048065
# Layer fc.5 multiply factor is 1548.0, normalized multiply factor is 1548
# Layer fc.8 max value is 0.09124133, normalized max value is 0.09124132990837097
# Layer fc.8 multiply factor is 1391.0, normalized multiply factor is 1391
