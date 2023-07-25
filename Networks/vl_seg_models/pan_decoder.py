import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_relu: bool = True,
        interpolate: bool = False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPABlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_mode="bilinear"):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == "bilinear":
            self.align_corners = True
        else:
            self.align_corners = False

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)

        x = torch.mul(x, mid)
        x = x + b1
        return x

    def merge_batchnorm_to_conv(self):

        def merge_conv_bn(conv, bn, linear=False):
            device = conv.weight.data.device
            if linear:
                new_conv = nn.Linear(int(conv.weight.size(1)), int(conv.weight.size(0)), bias=True).eval().to(device)
            else:
                new_conv = nn.Conv2d(int(conv.weight.size(1)) * conv.groups, int(conv.weight.size(0)),
                                     kernel_size=int(conv.weight.size(2)), stride=conv.stride, padding=conv.padding,
                                     groups=conv.groups, bias=True).eval().to(device)

            bn_scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
            bn_bias = bn.bias.data - bn.running_mean * bn_scale

            new_conv.bias = Parameter(bn_bias) if conv.bias is None else Parameter(bn_bias + conv.bias.data * bn_scale)
            weight_matrix = torch.stack([weight * val for weight, val in zip(conv.weight.data, bn_scale)])
            new_conv.weight = Parameter(weight_matrix)
            return new_conv

        child_list = [child for child in self.named_modules() if self._is_leaf(child[1])]
        interest_child_list = []
        for i_child, child in enumerate(child_list):
            if isinstance(child[1], nn.BatchNorm2d):
                interest_child_list.append([child_list[i_child - 1], child_list[i_child]])

        for conv_child, bn_child in interest_child_list:
            merged_conv = merge_conv_bn(conv_child[1], bn_child[1],
                                        linear=isinstance(conv_child[1], nn.Linear))
            net_conv = self._get_attribute(self, conv_child[0].split('.'))
            net_conv.bias = merged_conv.bias
            net_conv.weight = merged_conv.weight
            parent_module = self._get_attribute(self, bn_child[0].split('.')[:-1])
            print(f'merge {bn_child[0]} to {conv_child[0]}')
            delattr(parent_module, bn_child[0].split('.')[-1])

    @staticmethod
    def _is_leaf(model):
        def get_num_gen(gen):
            return sum(1 for _ in gen)

        return get_num_gen(model.children()) == 0

    @staticmethod
    def _get_attribute(object, attributes):
        out = object
        for attribute in attributes:
            out = getattr(out, attribute)
        return out

    def to_avx2(self):
        # merge bn, expand conv from 1 to 8 channels
        self.merge_batchnorm_to_conv()

        def conv4avx2(old_conv, change_in_ch=False):
            out_ch, in_ch, k_size = old_conv.out_channels, old_conv.in_channels, old_conv.kernel_size[0]
            assert out_ch == 1
            if change_in_ch:
                assert in_ch == 1
                new_in_ch = 8
            else:
                assert in_ch % 8 == 0
                new_in_ch = in_ch

            use_bias = old_conv.bias is not None
            new_conv = nn.Conv2d(new_in_ch, 8, k_size, old_conv.stride, old_conv.padding, bias=use_bias)
            new_conv = new_conv.to(old_conv.weight.data.device)
            new_conv.weight.data = torch.zeros_like(new_conv.weight.data)
            new_conv.weight.data[:out_ch, :in_ch] = old_conv.weight.data
            if use_bias:
                new_conv.bias.data = torch.zeros_like(new_conv.bias.data)
                new_conv.bias.data[:out_ch] = old_conv.bias.data
            return new_conv

        child_list = [child for child in self.named_modules() if self._is_leaf(child[1])][1:]
        filter_func = lambda child: isinstance(child[1], nn.Conv2d) and child[1].out_channels == 1
        for child in filter(filter_func, child_list):
            old_conv = self._get_attribute(self, child[0].split('.'))
            new_conv = conv4avx2(old_conv, old_conv.in_channels == 1)

            parent_module = self._get_attribute(self, child[0].split('.')[:-1])
            setattr(parent_module, child[0].split('.')[-1], new_conv)

        self.forward = self.forward_avx2

    def forward_avx2(self, x):
        b1 = self.branch1(x)
        mid = self.mid(x)
        # tensor for flower only
        x_zero = mid * 0
        index = torch.tensor(0, device=x.device)
        x_zero = x_zero.index_select(1, index)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x2 = self.conv2(x2)
        x3 = F.interpolate(x3, size=x2.size()[-2:], mode='bilinear')
        x = x2 + x3

        x1 = self.conv1(x1)
        x = F.interpolate(x, size=x1.size()[-2:], mode='bilinear')
        x = x + x1

        x = F.interpolate(x, size=x_zero.size()[-2:], mode='bilinear')
        x = x + x_zero

        x = x.index_select(1, index)
        x = torch.mul(x, mid)

        b1 = F.interpolate(b1, size=x.size()[-2:], mode='bilinear')
        x = x + b1
        return x


class GAUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_mode: str = "bilinear"):
        super(GAUBlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == "bilinear" else None

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                add_relu=False,
            ),
            nn.Sigmoid(),
        )
        self.conv2 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def to_avx2(self):
        self.forward = self.forward_avx2

    def forward_avx2(self, x, y):
        x = self.conv2(x)
        y1 = self.conv1(y)
        z = torch.mul(x, y1)

        y_up = F.interpolate(y, size=z.size()[-2:], mode='bilinear')
        return y_up + z

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PANDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, upscale_mode: str = "bilinear"):
        super().__init__()

        self.fpa = FPABlock(in_channels=encoder_channels[-1], out_channels=decoder_channels)
        self.gau3 = GAUBlock(
            in_channels=encoder_channels[-2],
            out_channels=decoder_channels,
            upscale_mode=upscale_mode,
        )
        self.gau2 = GAUBlock(
            in_channels=encoder_channels[-3],
            out_channels=decoder_channels,
            upscale_mode=upscale_mode,
        )
        self.gau1 = GAUBlock(
            in_channels=encoder_channels[-4],
            out_channels=decoder_channels,
            upscale_mode=upscale_mode,
        )

    def to_avx2(self):
        self.fpa.to_avx2()
        self.gau3.to_avx2()
        self.gau2.to_avx2()
        self.gau1.to_avx2()

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)  # 1/32
        x4 = self.gau3(features[-2], x5)  # 1/16
        x3 = self.gau2(features[-3], x4)  # 1/8
        x2 = self.gau1(features[-4], x3)  # 1/4

        return x2
