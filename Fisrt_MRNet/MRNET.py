import math
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.nn.functional as F

from arches import conv1x1, PredeblurModule


class FENet(nn.Layer):
    def __init__(self, n_features):
        super(FENet, self).__init__()
        self.init_feature = nn.Sequential(
            PredeblurModule(num_feat=n_features),
            nn.Conv2D(n_features, n_features, 1, 1, bias_attr=False),
        )

    def forward(self, x):
        out = self.init_feature(x)
        return out


class ResB(nn.Layer):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2D(channels, channels, 3, 1, 1, bias_attr=True),
            # nn.LeakyReLU(0.1),
            nn.PReLU(num_parameters=channels),
            nn.Conv2D(channels, channels, 3, 1, 1, bias_attr=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


class RG(nn.Layer):
    def __init__(self, n_features, n_resblocks=3):
        super(RG, self).__init__()

        modules_body = [
            ResB(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2D(n_features, n_features, 3, 1, 1, bias_attr=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RGM(nn.Layer):
    def __init__(self, n_features, n_resblocks=3):
        super(RGM, self).__init__()

        modules_body = [
            RG(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2D(n_features, n_features, 3, 1, 1, bias_attr=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DePixelShuffle(nn.Layer):
    def __init__(self, n_features, downscale_factor=2):
        super(DePixelShuffle, self).__init__()
        self.loop = int(math.log(downscale_factor, 2))
        self.tail = nn.Conv2D(n_features * downscale_factor ** 2, n_features, 1, 1, 0, bias_attr=False)

    def forward(self, x):
        for i in range(self.loop):
            d1 = x[:, :, ::2, ::2]
            d2 = x[:, :, 1::2, ::2]
            d3 = x[:, :, ::2, 1::2]
            d4 = x[:, :, 1::2, 1::2]
            # x = fluid.layers.concat(input=[d1, d2, d3, d4], axis=1)
            x = paddle.concat([d1, d2, d3, d4], axis=1)
        out = self.tail(x)
        return out


class UpPixelShuffle(nn.Layer):
    def __init__(self, n_features, upscale_factor=4):
        super(UpPixelShuffle, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2D(n_features, n_features * upscale_factor ** 2, 1, 1, 0, bias_attr=False),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return self.model(x)


class CROSSB(nn.Layer):
    def __init__(self, n_features):
        super(CROSSB, self).__init__()

        self.body1 = nn.Sequential(
            RGM(n_features),
        )

        self.body2 = nn.Sequential(
            DePixelShuffle(n_features, 2),
            RGM(n_features),
            UpPixelShuffle(n_features, 2),
        )
        self.body3 = nn.Sequential(
            DePixelShuffle(n_features, 4),
            RGM(n_features),
            UpPixelShuffle(n_features, 4),
        )

        self.fusion = conv1x1(n_features * 4, n_features)

    def forward(self, x):
        x1 = self.body1(x)
        x2 = self.body2(x)
        x3 = self.body3(x)

        # out = fluid.layers.concat(input=[x, x1, x2, x3], axis=1)
        out = paddle.concat([x, x1, x2, x3], axis=1)
        out = self.fusion(out)
        return out


class CROSS(nn.Layer):
    def __init__(self, n_features, n_resblocks=4):
        super(CROSS, self).__init__()

        modules_body = [
            CROSSB(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2D(n_features, n_features, 3, 1, 1, bias_attr=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Reconstructor
class Reconstruct(nn.Layer):
    def __init__(self, n_features, upscale_factor=4):
        super(Reconstruct, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2D(n_features, n_features * upscale_factor ** 2, 1, 1, 0, bias_attr=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2D(n_features, 3, 3, 1, 1, bias_attr=False),
            nn.Conv2D(3, 3, 3, 1, 1, bias_attr=False)

        )

    def forward(self, x):
        return self.model(x)


class MRNET(nn.Layer):
    def __init__(self, n_features=64):
        super(MRNET, self).__init__()

        self.n_features = n_features

        self.feature_extractor = FENet(n_features)
        self.fusion = conv1x1(n_features * 2, n_features)
        self.body = CROSS(n_features)
        self.reconstructor = Reconstruct(n_features)

    def forward(self, x):
        buffer_left = self.feature_extractor(x)
        buffer_right = self.feature_extractor(x)
        # buffer_tmp_f = fluid.layers.concat(input=[buffer_left, buffer_right], axis=1)
        buffer_tmp_f = paddle.concat([buffer_left, buffer_right], axis=1)
        buffer_tmp_f = self.fusion(buffer_tmp_f)
        buffer_tmp = self.body(buffer_tmp_f)
        outputs = self.reconstructor(buffer_tmp)
        outputs = outputs + x

        return outputs
