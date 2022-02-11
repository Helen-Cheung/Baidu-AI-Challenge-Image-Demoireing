import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResidualBlockNoBN(nn.Layer):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=True)
        self.conv2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PredeblurModule(nn.Layer):
    def __init__(self, num_in_ch=3, num_feat=64, hr_in=True):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2D(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2D(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2D(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2D(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2D(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.LayerList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def __call__(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias_attr=True)
