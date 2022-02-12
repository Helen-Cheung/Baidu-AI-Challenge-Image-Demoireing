import paddle
import functools
import paddle.nn as nn
import paddle.nn.functional as F


class DownSample(nn.Layer):
    def __init__(self, in_channels, mid_channels):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2D(mid_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        out = self.lrelu(self.bn(self.conv(x)))
        return out


class PixelShufflePack(nn.Layer):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2D(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ConvModule(nn.Layer):
    def __init__(self, in_channels, mid_channels):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2D(in_channels, mid_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2D(mid_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self,x):
        out = self.lrelu(self.bn(self.conv(x)))
        return out


class SpatialAttentionExtractor(nn.Layer):
    def __init__(self, in_channels=64, mid_channels=64):
        super(SpatialAttentionExtractor, self).__init__()

        # embedding 
        self.embedding = nn.Conv2D(in_channels, mid_channels, 3, 1, 1)

        # extractor
        self.feat_extractor = nn.Conv2D(in_channels, mid_channels, 3, 1, 1)

        # spatial attention
        self.max_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        self.spatial_attn = ConvModule(mid_channels * 2, mid_channels)
        
        # spatial attention add part
        self.spatial_attn_add1 = ConvModule(mid_channels, mid_channels)
        self.spatial_attn_add2 = ConvModule(mid_channels, mid_channels)

        # umsample
        self.upsample = PixelShufflePack(mid_channels, mid_channels, 2, 3)

        # activate 
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()
    
    def spatial_padding(self, x):
        _, c, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        
        # padding
        lrs = x.reshape_((-1, c, h, w))
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode="reflect")

        return lrs

    def forward(self, x):
        _, _, h_input, w_input = x.shape
        # x = self.spatial_padding(x)

        # extract feature
        feat = self.embedding(x)

        # spatial attention
        attn = self.embedding(x)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn(paddle.concat([attn_max, attn_avg], axis=1))
        attn = self.upsample(attn)
        attn = self.sigmoid(attn)
        
        # spatial attention add
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))

        out = feat * attn * 2 + attn_add
        return out[:, :, :h_input, :w_input]


def make_multi_blocks(func, num_layers):
    """Make layers by stacking the same blocks.

    Args:
        func (nn.Layer): nn.Layer class for basic block.
        num_layers (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    Blocks = nn.Sequential()
    for i in range(num_layers):
        Blocks.add_sublayer('block%d' % i, func())
    return Blocks


def make_multi_dilation_blocks(mid_channels, num_blocks=5):
    dilation_list = [1,2,5,7,12]
    layers = []
    for idx in range(num_blocks):
        layers.append(SCPA(nf=mid_channels, dilation=dilation_list[idx]))
    
    return nn.Sequential(*layers)


class PA(nn.Layer):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2D(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y

        return out


class PAConv(nn.Layer):
    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2D(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution
        self.k4 = nn.Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = self.k3(x) * y
        out = self.k4(out)

        return out


class SCPA(nn.Layer):
    """
    SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2D(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)
        self.conv1_b = nn.Conv2D(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)

        self.k1 = nn.Sequential(
            nn.Conv2D(group_width,
                      group_width,
                      kernel_size=3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      bias_attr=False))

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2D(group_width * reduction,
                               nf,
                               kernel_size=1,
                               bias_attr=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(paddle.concat([out_a, out_b], axis=1))
        out += residual

        return out


class PAN(nn.Layer):
    def __init__(self, in_nc, out_nc, nf, unf, nb):
        super(PAN, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)

        ### first convolution
        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1)

        ### main blocks
        self.SCPA_trunk = make_multi_blocks(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2D(nf, nf, 3, 1, 1)

        #### upsampling
        self.upconv1 = nn.Conv2D(nf, unf, 3, 1, 1)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2D(unf, unf, 3, 1, 1)

        self.att2 = PA(unf)
        self.HRconv2 = nn.Conv2D(unf, unf, 3, 1, 1)

        self.att3 = PA(unf)
        self.HRconv3 = nn.Conv2D(unf, unf, 3, 1, 1)


        self.conv_last = nn.Conv2D(unf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        
        fea = self.relu(self.att2(fea))
        fea = self.relu(self.HRconv2(fea))
        fea = self.relu(self.att3(fea))
        fea = self.relu(self.HRconv3(fea))

        out = self.conv_last(fea)
        out = out + x
        return out
