'''
使用paibox进行仿真，对paf_filter数据集进行推理'''
import torch
import numpy as np
import paibox as pb
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import functional
from DVS_dataset import DVSDataset
from quantize.qat_model import PAIBOX_DVSNet_FC, QAT_DVSNet
from torchvision import transforms
import torch.nn.functional as F


def getNetParam(param_dict, timestep, delay):
    # layer_num = sum(1 for key in param_dict["model"] if "weight" in key)
    # delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["delay"] = delay
    param_dict["vthr"] = 127
    return param_dict


# def pb_infer_sim(test_data_loader, param_dict, pb_net, feature_extactor):
#     sim = pb.Simulator(pb_net)
#     test_acc_pb = 0
#     test_acc_sj = 0
#     test_samples = 0
#     feature_extactor.eval()
#     with torch.no_grad():
#         for i, (image_tensor, label_tensor) in enumerate(test_data_loader):
#             total_sj_out = np.empty((0, 11))
#             # total_pb_out = np.array([])
#             if i == 50:
#                 break
#             label = label_tensor[0]
#             test_samples += 1
#             label = label.item()
#             image_withT = (
#                 torch.tensor(np.where(image_tensor.detach().numpy() == 0, 0, 1).astype(np.int8))
#                 .float()
#                 .transpose(0, 1)  #
#             )
#             for t, image in enumerate(image_withT):
#                 image_feature, sj_out = feature_extactor(image)
#                 # sj
#                 total_sj_out = np.append(total_sj_out, sj_out, axis=0)
#                 # pb
#                 image_feature = image_feature.numpy().astype(np.uint8)
#                 image_feature = image_feature.squeeze()
#                 pb_net.inp.input = image_feature
#                 sim.run(1 + (0 if t + 1 != param_dict["timestep"] else param_dict["delay"]))

#             total_pb_out = sim.data[pb_net.probe1].astype(np.float32)
#             total_pb_out = total_pb_out[param_dict["delay"] :]
#             total_pb_out = torch.tensor(total_pb_out).unsqueeze(1)
#             vote = layer.VotingLayer(10, step_mode="m")
#             total_pb_out = vote(total_pb_out).mean(0)
#             test_acc_pb += total_pb_out.argmax(1).item() == label

#             total_sj_out = total_sj_out.mean(0)
#             sj_out = total_sj_out.argmax().item()
#             test_acc_sj += sj_out == label

#             functional.reset_net(feature_extactor)
#             sim.reset()
#         test_acc_sj /= test_samples
#         test_acc_pb /= test_samples
#         print("test_acc_pb: ", test_acc_pb)
#         print("test_acc_sj: ", test_acc_sj)


def pb_infer_sim(test_data_loader, param_dict, pb_net, origin_model, device):
    sim = pb.Simulator(pb_net)
    test_samples = 0
    total_pb_out = np.array([])
    test_acc_pb = 0
    test_acc_sj = 0
    origin_model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_data_loader):
            test_samples += 1
            label = label[0]
            label = label.item()
            N, T, C, H, W = image.shape
            resize = transforms.Resize((128, 128))
            resized_tensor = torch.stack([resize(image[n, t, :, :, :]) for n in range(N) for t in range(T)])
            # 重新调整张量形状为 (N, T, C, 128, 128)
            image_resize = resized_tensor.view(N, T, C, 128, 128)
            image_resize = image_resize.transpose(0, 1)
            image_resize.to(device, non_blocking=True)

            feat, sj_out = origin_model(image_resize)
            for t, feature in enumerate(feat):
                feature = feature.squeeze()  # torch.Size([1024])
                feature = feature.numpy().astype(np.uint8)
                pb_net.inp.input = feature

                sim.run(1 + (0 if t + 1 != param_dict["timestep"] else param_dict["delay"]))

            total_pb_out = sim.data[pb_net.probe1].astype(np.float32)
            total_pb_out = total_pb_out[param_dict["delay"] :]
            total_pb_out = torch.tensor(total_pb_out).unsqueeze(1)
            
            fc_out = origin_model.fc(total_pb_out)
            out = origin_model.head(fc_out.mean(0))
            
            test_acc_pb += out.argmax(1).item() == label
            test_acc_sj += sj_out.argmax(1).item() == label
            sim.reset()
            functional.reset_net(origin_model)
        test_acc_pb /= test_samples
        test_acc_sj /= test_samples
        print("test_acc_sj: ", test_acc_sj)
        print("test_acc_pb: ", test_acc_pb)


def pb_infer_deploy():
    pass

def load_data_from_dvs():
    pass

def load_data(data_dir=None, sim=True, T=16):
    if sim and data_dir:
        dataset_test = DVSDataset(root=data_dir, train=True, data_type="frame", frames_number=T, split_by="number")
        return dataset_test
    else:
        pass


def main():
    data_dir = "/home/zdm/dataset/PAF_filter"
    channel = 32
    T = 16
    w_bits = 8
    step_mode = "m"
    checkpoint_path = "./QAT_DVSNet_best.pth"
    device = "cuda:0"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    param_dict = getNetParam(checkpoint, T, 1)
    origin_model = QAT_DVSNet(w_bits=w_bits, channels=channel, n_cls=9)
    origin_model.load_state_dict(checkpoint["model"])
    pb_net = PAIBOX_DVSNet_FC(inp_shape=(256,), param_dict=param_dict, w_bits=w_bits)
    
    mapper = pb.Mapper()
    mapper.build(pb_net)
    graph_info = mapper.compile(core_estimate_only=False, weight_bit_optimization=True, grouping_optim_target="both")
    mapper.export(write_to_file=True, fp="./debug/", format="bin", split_by_chip=False, use_hw_sim=True)


    dataset_test = load_data(data_dir)
    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
    )
    device = torch.device(device)
    functional.set_step_mode(origin_model, step_mode)
    pb_infer_sim(data_loader_test, param_dict, pb_net, origin_model, device)


if __name__ == "__main__":
    main()
