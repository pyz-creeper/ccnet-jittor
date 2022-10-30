import jittor as jt
from jittor import nn

def NEG_INF_DIAG(n: int):
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return jt.diag(jt.Var(float('-inf')).repeat(n), 0)

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(jt.Var(scale).float32())

    def execute(self, x):
        return x * self.scale

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def execute(self,x):
        B, C, H, W = x.shape
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = jt.linalg.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(
            H)
        energy_H = energy_H.transpose(1, 2)
        energy_W = jt.linalg.einsum('bchw,bchj->bhwj', query, key)
        attn = jt.nn.softmax(
            jt.concat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]
        out = jt.linalg.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += jt.linalg.einsum('bchj,bhwj->bchw', value, attn[..., H:])

        out = self.gamma(out) + x
        return out

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s
