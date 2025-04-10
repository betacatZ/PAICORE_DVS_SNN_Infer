import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, surrogate, functional, base
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
import numpy as np


class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class WeightQuantizer(nn.Module):
    def __init__(self, w_bits, infer=False):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.infer = infer

    # 取整(ste)
    @staticmethod
    def round(input):
        output = Round.apply(input)
        return output

    @staticmethod
    def quantize(input, w_bits):
        output = torch.tanh(input)
        output = output / torch.max(torch.abs(output))
        # print("tanh", output)
        scale = 1 / float(2 ** (w_bits - 1) - 1)  # scale
        output = WeightQuantizer.round(output / scale)
        return output, scale

    # 量化/反量化
    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print("Binary quantization is not supported")
            assert self.w_bits != 1
        else:
            # output = torch.tanh(input)
            # output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            # scale = 1 / float(2 ** self.w_bits - 1)  # scale
            # output = self.round(output / scale) * scale  # 量化/反量化
            # output = 2 * output - 1

            # output = torch.tanh(input)
            # output = output / torch.max(torch.abs(output))
            # # print("tanh", output)
            # scale = 1 / float(2 ** (self.w_bits - 1) - 1)  # scale
            # # print("量化", self.round(output / scale))

            output, scale = WeightQuantizer.quantize(input, self.w_bits) # 量化
            if self.infer:
                output = output * scale  # 反量化
            return output

    def extra_repr(self):
        return super().extra_repr() + f", w_bits={self.w_bits}"


class Quan_Linear(nn.Linear, base.StepModule):
    def __init__(
        self, w_bits: int, in_features: int, out_features: int, bias: bool = True, step_mode: str = "s", infer=False
    ):
        super(Quan_Linear, self).__init__(in_features, out_features, bias)
        self.step_mode = step_mode
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits, infer=infer)
        self.bias_quantizer = WeightQuantizer(w_bits=w_bits, infer=infer)

    def forward(self, x: torch.Tensor):
        quan_weight = self.weight_quantizer(self.weight)
        if self.bias is not None:
            quan_bias = self.bias_quantizer(self.bias)
        else:
            quan_bias = self.bias
        return F.linear(x, quan_weight, quan_bias)


class Quan_Conv1d(nn.Conv1d, base.StepModule):
    def __init__(
        self,
        w_bits: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ):
        super(Quan_Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)
        self.bias_quantizer = WeightQuantizer(w_bits=w_bits)

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor):
        quan_weight = self.weight_quantizer(self.weight)
        if self.bias is not None:
            quan_bias = self.bias_quantizer(self.bias)
        else:
            quan_bias = self.bias

        if self.step_mode == "s":
            x = F.conv1d(x, quan_weight, quan_bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.step_mode == "m":
            if x.dim() != 5:
                raise ValueError(f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!")
            y_shape = [x.shape[0], x.shape[1]]
            y = x.flatten(0, 1)
            y = F.conv1d(y, quan_weight, quan_bias, self.stride, self.padding, self.dilation, self.groups)
            y_shape.extend(y.shape[1:])
            return y.view(y_shape)

        return x


class Quan_Conv2d(nn.Conv2d, base.StepModule):
    def __init__(
        self,
        w_bits: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ):
        super(Quan_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor):
        quan_weight = self.weight_quantizer(self.weight)
        if self.step_mode == "s":
            x = F.conv2d(x, quan_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        elif self.step_mode == "m":
            if x.dim() != 5:
                raise ValueError(f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!")
            y_shape = [x.shape[0], x.shape[1]]
            y = x.flatten(0, 1)
            y = F.conv2d(y, quan_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            y_shape.extend(y.shape[1:])
            return y.view(y_shape)

        return x
