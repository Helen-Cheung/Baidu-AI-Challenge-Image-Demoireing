import torch
import torch.nn as nn
import torch.nn.functional as F
from archestorch import conv1x1, PredeblurModule
import math


class FENet(nn.Module):
    def __init__(self, n_features):
        super(FENet, self).__init__()
        self.init_feature = nn.Sequential(
            PredeblurModule(num_feat=n_features),
            nn.Conv2d(n_features, n_features, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.init_feature(x)
        return out


# Reconstructor
class Reconstruct(nn.Module):
    def __init__(self, n_features, upscale_factor=4):
        super(Reconstruct, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_features, n_features * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(n_features, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)

        )

    def forward(self, x):
        return self.model(x)


## Residual Block (RB)
class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.PReLU(num_parameters=channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


## Residual Group (RG)
class RG(nn.Module):
    def __init__(self, n_features, n_resblocks=3):
        super(RG, self).__init__()

        modules_body = [
            ResB(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_features, n_features, 3, 1, 1, bias=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class RGB(nn.Module):
    def __init__(self, n_features, n_resblocks=3):
        super(RGB, self).__init__()

        modules_body = [
            RG(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_features, n_features, 3, 1, 1, bias=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DePixelShuffle(nn.Module):
    def __init__(self, n_features, downscale_factor=2):
        super(DePixelShuffle, self).__init__()
        self.loop = int(math.log(downscale_factor, 2))
        self.tail = nn.Conv2d(n_features * downscale_factor ** 2, n_features, 1, 1, 0, bias=False)

    def forward(self, x):
        for i in range(self.loop):
            d1 = x[:, :, ::2, ::2]
            d2 = x[:, :, 1::2, ::2]
            d3 = x[:, :, ::2, 1::2]
            d4 = x[:, :, 1::2, 1::2]
            x = torch.cat([d1, d2, d3, d4], 1)
        out = self.tail(x)
        return out


class UpPixelShuffle(nn.Module):
    def __init__(self, n_features, upscale_factor=4):
        super(UpPixelShuffle, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_features, n_features * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return self.model(x)


class CROSSB(nn.Module):
    def __init__(self, n_features):
        super(CROSSB, self).__init__()

        self.body1 = nn.Sequential(
            RGB(n_features),
        )

        self.body2 = nn.Sequential(
            DePixelShuffle(n_features, 2),
            RGB(n_features),
            UpPixelShuffle(n_features, 2),
        )
        self.body3 = nn.Sequential(
            DePixelShuffle(n_features, 4),
            RGB(n_features),
            UpPixelShuffle(n_features, 4),
        )

        self.fusion = conv1x1(n_features * 4, n_features)

    def forward(self, x):
        x1 = self.body1(x)
        x2 = self.body2(x)
        x3 = self.body3(x)

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.fusion(out)
        return out


class CROSS(nn.Module):
    def __init__(self, n_features, n_resblocks=4):
        super(CROSS, self).__init__()

        modules_body = [
            CROSSB(n_features) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_features, n_features, 3, 1, 1, bias=False))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class MRNET_TORCH(nn.Module):
    def __init__(self, n_features=64):
        super(MRNET_TORCH, self).__init__()
        self.feature_extractor = FENet(n_features)
        self.fusion = conv1x1(n_features * 2, n_features)
        self.body = CROSS(n_features)
        self.reconstructor = Reconstruct(n_features)

    def forward(self, x):
        # feature_extractor
        buffer_left = self.feature_extractor(x)
        buffer_right = self.feature_extractor(x)

        # fusion
        buffer_tmp_f = torch.cat([buffer_left, buffer_right], dim=1)
        buffer_tmp_f = self.fusion(buffer_tmp_f)  # (N, C, H/4, W/4)
        buffer_tmp = self.body(buffer_tmp_f)

        # Reconstructor
        outputs_tmp = self.reconstructor(buffer_tmp)
        outputs = outputs_tmp + x

        return outputs
