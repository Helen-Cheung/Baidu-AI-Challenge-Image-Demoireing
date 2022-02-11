import paddle
import paddle.nn as nn


class L1_Charbonnier_loss(nn.Layer):
    """
    L1 Charbonnierloss
    """

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = paddle.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = paddle.mean(diff_sq, 1, True)
        error = paddle.sqrt(diff_sq_color + self.eps * self.eps)
        loss = paddle.mean(error)
        return loss
