class C2G(nn.Module):
    def __init__(self, channel):
        super(C2G, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CG = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.max_pool(x)
        max_out_1 = self.CG(max_out)
        max_out_1 = max_out_1*x

        avg_out = self.avg_pool(x)
        avg_out_1 = self.conv5(avg_out)
        avg_out_1 = avg_out_1*x

        out = self.sigmoid(avg_out_1 + max_out_1)
        out = out * x
        return out


class unetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp1, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)

        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.Gelu   = nn.GELU()

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.Gelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.Gelu(outputs)
        return outputs
# 最终改进
class Unet1(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet1, self).__init__()
        # 选择主干vgg
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        self.ca4 = C2G(out_filters[3])
        self.ca3 = C2G(out_filters[2])
        self.ca2 = C2G(out_filters[1])
        self.ca1 = C2G(out_filters[0])
        # 上采样使用强化特征提取和特征学习
        # 64,64,512
        self.up_concat4 = unetUp1(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp1(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp1(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp1(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        feat4 = self.ca4(feat4)
        feat3 = self.ca3(feat3)
        feat2 = self.ca2(feat2)
        feat1 = self.ca1(feat1)
        # 上采样使用强化特征提取和特征学习
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False


    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True