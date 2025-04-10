from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, surrogate, neuron, layer
import paibox as pb
# from timm.models.registry import register_model
from .qat_quantize_layer import Quan_Conv2d, Quan_Linear, WeightQuantizer


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000, bias=True
    ):
        super().__init__()
        self.add_module("c", layer.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=bias))
        self.add_module("bn", layer.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5  # type: ignore
        m = layer.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class QAT_DVSNet(nn.Module):
    def create_conv_block(self, in_channels, out_channels, ks=3, pad=1, stride=1, vthr=None):
        """
        Helper function to create a convolution block with batch normalization, convolution, and neuron.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            ks (int): Kernel size. Default is 3.
            pad (int): Padding. Default is 1.
            stride (int): Stride. Default is 1.
            vthr (float or None): Optional threshold for neuron. Default is None.

        Returns:
            nn.Module: A Sequential block of Conv2d, BatchNorm, and Neuron (IFNode).
        """
        conv_block = [
            Conv2d_BN(in_channels, out_channels, ks=ks, pad=pad, stride=stride, bias=False),
            neuron.IFNode(detach_reset=True),
        ]

        # if vthr is not None:
        #     conv_block[1] = neuron.LIFNode(detach_reset=True, v_threshold=float(vthr))

        return nn.Sequential(*conv_block)

    def __init__(self, w_bits, channels: int, vthr=1.0, n_cls=11, step_mode="m", **kwargs):
        super().__init__()
        conv = []
        conv.extend(
            [
                self.create_conv_block(2, channels),
                self.create_conv_block(channels, channels * 2, ks=2, pad=0, stride=2),
                self.create_conv_block(channels * 2, channels * 2),
                self.create_conv_block(channels * 2, channels * 4, ks=2, pad=0, stride=2),
                self.create_conv_block(channels * 4, channels * 4),
                self.create_conv_block(channels * 4, channels * 4, ks=2, pad=0, stride=2),
                self.create_conv_block(channels * 4, channels // 8),
                # self.create_conv_block(channels * 8, channels * 8, ks=2, pad=0, stride=2),
            ]
        )
        self.conv = nn.Sequential(*conv)
        self.flatten = layer.Flatten()
        self.pool = nn.Sequential(layer.MaxPool2d(2, 2))
        snn_fc = []

        snn_fc.append(layer.Dropout(0.5))
        snn_fc.append(Quan_Linear(w_bits, 256, 256, bias=False, infer=True))
        # snn_fc.append(layer.Linear(1024, 256, bias=False))
        snn_fc.append(neuron.IFNode(detach_reset=True, v_threshold=vthr))

        self.snn_fc = nn.Sequential(*snn_fc)
        fc = []
        # fc.append(Quan_Linear(w_bits, 256, 176, bias=False))
        fc.append(layer.Linear(256, 176, bias=False))

        fc.append(layer.Dropout(0.5))
        self.fc = nn.Sequential(*fc)
        # self.head = Quan_Linear(w_bits, 176, n_cls, bias=False)
        self.head = layer.Linear(176, n_cls, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        conv_feature = x
        out_spikes = self.snn_fc(x)
        out = self.fc(out_spikes)
        return conv_feature, self.head(out.mean(0))


class PAIBOX_DVSNet_FC(pb.Network):
    def __init__(self, inp_shape, param_dict, w_bits):
        super().__init__()
        self.inp = pb.InputProj(input=None, shape_out=inp_shape)
        fc_layers = self.get_layer_list(param_dict, key="snn_fc")
        tick_wait_start = 1
        fc_layers_config = [
            {
                "name": "fc_1",
                "n": "n_fc_1",
                "out_size": 256,
            },
        ]
        prev_fc = self.inp
        for i, snn_layer in enumerate(fc_layers):
            print(snn_layer)
            fc_n = pb.LIF(
                fc_layers_config[i]["out_size"],
                threshold=param_dict["vthr"],
                reset_v=0,
                tick_wait_start=tick_wait_start,
            )
            setattr(self, fc_layers_config[i]["n"], fc_n)

            weights, _ = WeightQuantizer.quantize(param_dict["model"][snn_layer + ".weight"], w_bits)
            weights = weights.T.to(torch.int8)
            fc_conn = pb.FullConn(
                prev_fc,
                fc_n,
                conn_type=pb.SynConnType.All2All,
                weights=weights,
            )
            setattr(self, fc_layers_config[i]["name"], fc_conn)
            prev_fc: pb.LIF = fc_n
        self.probe1 = pb.Probe(self.n_fc_1, "spike")

    def get_layer_list(self, param_dict, key=None):
        layer_name_list = list(param_dict["model"].keys())
        layer_list = []
        seen = set()
        for name in layer_name_list:
            layer_name = name.rsplit(".", 1)[0]
            if layer_name not in seen:
                layer_list.append(layer_name)
                seen.add(layer_name)
        if key:
            layer_list = [elem for elem in layer_list if key in elem]
        return layer_list


# @register_model
# def qat_dvsnet(
#     pretrained=False,
#     pretrained_cfg=False,
#     checkpoint=None,
#     w_bits=8,
#     channels=32,
#     vthr=1.0,
#     n_cls=11,
#     step_mode="m",
#     **kwargs,
# ):
#     model = QAT_DVSNet(w_bits=w_bits, channels=channels, vthr=vthr, n_cls=n_cls)
#     if pretrained and checkpoint is not None:
#         model.load_state_dict(checkpoint)
#     functional.set_step_mode(model, step_mode)

#     return model


def test_sj_net():
    x = torch.rand([16, 3, 2, 128, 128])  # [T, N, C, H, W]
    net = QAT_DVSNet(w_bits=8, channels=32, vthr=1.0, n_cls=9)
    print("\nState Dict:")
    for name, param in net.state_dict().items():
        print(f"{name}: {param.shape if param.requires_grad else 'Non-trainable'}")
    functional.reset_net(net)
    functional.set_step_mode(net, "m")
    print(net)
    print(net(x)[1].shape)
    from torchinfo import summary

    summary(net, input_size=(16, 3, 2, 128, 128))


def show_qat_dvsnet_weight():
    net = QAT_DVSNet(w_bits=8, channels=32, vthr=1.0, n_cls=11)
    checkpoint = torch.load(
        "/home/zdm/code/PEFD/save/student_model/S:QAT_DVSNet_T:EfficientSFormer_S_64_120_176_dvs128_ours_r:0.5_a:0_b:0.5_1_gelu/QAT_DVSNet_best.pth"
    )
    net.load_state_dict(checkpoint["model"])

    # 量化三个不同的权重
    output1, _ = WeightQuantizer.quantize(net.fc[1].weight, 8)
    output2, _ = WeightQuantizer.quantize(net.fc[3].weight, 8)
    output3, _ = WeightQuantizer.quantize(net.head.weight, 8)

    # 转换为NumPy数组
    data1 = output1.detach().numpy()
    data2 = output2.detach().numpy()
    data3 = output3.detach().numpy()

    # 创建有3个子图的图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制第一个子图 - fc[1].weight
    axes[0].hist(data1, bins=50, range=(-127, 127), edgecolor="navy", alpha=0.7)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("FC Layer 1 Weight Distribution")
    axes[0].grid(True)

    # 绘制第二个子图 - fc[2].weight
    axes[1].hist(data2, bins=50, range=(-127, 127), edgecolor="darkgreen", alpha=0.7)
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("FC Layer 2 Weight Distribution")
    axes[1].grid(True)

    # 绘制第三个子图 - head.weight
    axes[2].hist(data3, bins=50, range=(-127, 127), edgecolor="darkred", alpha=0.7)
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Head Layer Weight Distribution")
    axes[2].grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图像到PNG文件
    plt.savefig("weight_distributions.png", dpi=300, bbox_inches="tight")

    print("图像已保存到 weight_distributions.png")


# if __name__ == "__main__":
#     test_sj_net()
