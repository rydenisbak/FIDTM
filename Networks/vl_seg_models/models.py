import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.decoders.pan.model import PANDecoder
from Networks.vl_seg_models.pan_decoder import PANDecoder


class VLSegModel(nn.Module):
    def __init__(self, args):
        super(VLSegModel, self).__init__()

        self.encoder = smp.encoders.get_encoder(
            args['model_type'][3:],
            in_channels=3,
            depth=5,
            weights='imagenet',
            output_stride=16,
        )

        decoder_channels = args.get('decoder_channels', 64)
        self.decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        head_type = args.get('head_type', 'hrnet')
        if head_type == 'hrnet':
            self.last_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_channels * 4,
                    out_channels=decoder_channels * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.BatchNorm2d(decoder_channels * 4, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(decoder_channels * 4, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),

            )
        elif head_type == 'flower':
            self.kostyl_transition_conv = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=False)
            self.last_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_channels * 4,
                    out_channels=decoder_channels * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.BatchNorm2d(decoder_channels * 4, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            stem_out_ch = next(self.encoder.children()).out_channels
            self.last_layer2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_channels * 4 + stem_out_ch,
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.BatchNorm2d(64, momentum=0.01),
                nn.ReLU(inplace=True)
            )

            self.last_layer3 = nn.Conv2d(64 + 8, 1, kernel_size=1, stride=1, padding=0)

        self.head_type = head_type
        self.initialize_decoder()
        self.initialize_head()
        self._is_avx2 = False

    def initialize_decoder(self):
        for m in self.decoder.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self):
        if self.head_type == 'hrnet':
            for m in self.last_layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            for m in self.last_layer1.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            for m in self.last_layer2.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            for m in self.last_layer3.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def decoder_forward(self, *features):
        decoder = self.decoder
        bottleneck = features[-1]
        x5 = decoder.fpa(bottleneck)  # 1/32
        x4 = decoder.gau3(features[-2], x5)  # 1/16
        x3 = decoder.gau2(features[-3], x4)  # 1/8
        x2 = decoder.gau1(features[-4], x3)  # 1/4

        return [x2, x3, x4, x5]

    @staticmethod
    def _is_leaf(model):
        def get_num_gen(gen):
            return sum(1 for _ in gen)

        return get_num_gen(model.children()) == 0

    def merge_normalize(self, mean=torch.Tensor([123.675, 116.28, 103.53]), std=torch.Tensor([58.395, 57.12, 57.375])):
        key, conv = [child for child in self.named_modules() if self._is_leaf(child[1])][0]

        weight = conv.weight.data
        std, mean = std.to(weight.device), mean.to(weight.device)
        weight = weight.permute(0, 2, 3, 1) / std
        conv.weight.data = weight.permute(0, 3, 1, 2)

        b_add = []
        for w in weight:
            to_sum = (-w * mean).reshape(-1)
            to_sum_sort_idx = torch.sort(torch.abs(to_sum))[1]
            b_add.append(torch.sum(to_sum[to_sum_sort_idx]))

        b_add = torch.tensor(b_add).to(weight.device)
        if conv.bias is not None:
            conv.bias.data += b_add.float()
        else:
            conv.bias = nn.Parameter(b_add.float())

    def to_avx2(self):
        if not self._is_avx2:
            self.decoder.to_avx2()
            self.forward = self.forward_avx2
            self._is_avx2 = True

    def forward_avx2(self, x):
        x0, x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.decoder.fpa(x5)
        x4 = self.decoder.gau3(x4, x5)
        x3 = self.decoder.gau2(x3, x4)
        x2 = self.decoder.gau1(x2, x3)

        x3 = F.interpolate(x3, x2.size()[-2:], mode='bilinear')
        x2 = torch.cat([x2, x3], dim=1)

        x4 = F.interpolate(x4, x2.size()[-2:], mode='bilinear')
        x2 = torch.cat([x2, x4], dim=1)

        x5 = F.interpolate(x5, x2.size()[-2:], mode='bilinear')
        x2 = torch.cat([x2, x5], dim=1)

        x2 = F.interpolate(self.last_layer1(x2), x1.size()[-2:], mode='bilinear')
        x1 = self.last_layer2(torch.cat([x2, x1], dim=1))

        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear')
        x0 = torch.cat([x1, self.kostyl_transition_conv(x0)], 1)

        return self.last_layer3(x0)

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_features = self.decoder_forward(*encoder_features)

        x0_h, x0_w = decoder_features[0].size(2), decoder_features[0].size(3)
        features = [F.upsample(item, size=(x0_h, x0_w), mode='bilinear') for item in decoder_features[1:]]
        feature = torch.cat([decoder_features[0]] + features, dim=1)
        if self.head_type == 'hrnet':
            out = self.last_layer(feature)
        else:
            feature = F.upsample(self.last_layer1(feature), encoder_features[1].size()[-2:], mode='bilinear')
            feature = self.last_layer2(torch.cat([feature, encoder_features[1]], 1))
            feature = F.upsample(feature, encoder_features[0].size()[-2:], mode='bilinear')
            feature = torch.cat([feature, self.kostyl_transition_conv(encoder_features[0])], 1)
            out = self.last_layer3(feature)

        return out


if __name__ == '__main__':
    from mmcv.cnn.utils import get_model_complexity_info
    out_onnx_file = '/Users/den/Downloads/crowd_regnetx_mn_opset9.onnx'
    torch_weights_file = '/Users/den/Downloads/MIX01/model_best.pth'

    model = VLSegModel({'model_type': 'vl_timm-regnetx_064', 'head_type': 'flower'})
    sd = torch.load(torch_weights_file, 'cpu')['state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    model.to_avx2()
    y2 = model(x)
    print(torch.mean(torch.abs(y - y2)))
    model.merge_normalize()

    input_shape = 1, 3, 640, 640
    feat = torch.randn(input_shape)
    torch.onnx.export(model, feat, out_onnx_file, verbose=True, opset_version=9)
    flops, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
    print(f'flops: {flops}, params: {params}')

    import onnx
    model = onnx.load(out_onnx_file)
    graph_def = model.graph
    nodes = graph_def.node

    for node in [node for node in nodes if node.op_type == 'Upsample']:
        node.attribute[0].s = b'bilinear'

    onnx.checker.check_model(model)
    onnx.save(model, out_onnx_file)

    # arm version (dilation = 1)

    model = onnx.load(out_onnx_file)
    graph_def = model.graph
    nodes = graph_def.node
    for node in nodes:
        if not hasattr(node, 'attribute'):
            continue
        for attribute in node.attribute:
            if not hasattr(attribute, 'name'):
                continue
            if attribute.name != 'dilations':
                continue
            assert len(attribute.ints) == 2 and attribute.ints[0] == attribute.ints[1], str(node)
            if attribute.ints[0] != 1:
                print(node.name)
                print(attribute)
                print(node.attribute[3])
                attribute.ints[0] = attribute.ints[1] = 1
                if node.attribute[3].ints[0] == 2:
                    node.attribute[3].ints[0] = node.attribute[3].ints[1] = node.attribute[3].ints[2] = \
                        node.attribute[3].ints[3] = 1

            break

    onnx.checker.check_model(model)
    out_onnx_file = out_onnx_file.replace('.onnx', '_arm.onnx')
    onnx.save(model, out_onnx_file)


