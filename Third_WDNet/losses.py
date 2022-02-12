import paddle
import paddle.nn as nn
import vgg


def compute_l1_loss(input, output):
    return paddle.mean(paddle.abs(input - output))


def compute_charnonnier_loss(input, output, eps=1e-12):
    return paddle.mean(paddle.sqrt((input - output)** 2 + eps))


def loss_textures(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.reshape([x.shape[0], -1, nc, x.shape[2], x.shape[3]])
    yi = y.reshape([y.shape[0], -1, nc, y.shape[2], y.shape[3]])

    xi2 = paddle.sum(xi * xi, axis=2)
    yi2 = paddle.sum(yi * yi, axis=2)
    out = paddle.nn.functional.relu(yi2 * alpha - xi2 + margin)

    return paddle.mean(out)


class LossNetwork(nn.Layer):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self, pretrained: str = None):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=pretrained).features

        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",  # 1_2 to 5_2
        }

    def forward(self, x):
        output = {}
        # import pdb
        # pdb.set_trace()
        for name, module in self.vgg_layers._sub_layers.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x

        return output



class TVLoss(nn.Layer):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = paddle.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = paddle.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.shape[1] * t.shape[2] * t.shape[3]

if __name__ == '__main__':
    img = paddle.randn([1, 3, 224, 224])

    net = LossNetwork()

    out = net(img)