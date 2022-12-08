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
        energy_H = jt.linalg.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(H)
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


class DilatedCrissCrossAttention(nn.Module):
    def __init__(self, in_channels: int, dilated: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels
        self.dilated = dilated

    def execute(self,x):
        B, C, H, W = x.shape

        # calculate dilated mask
        h_mask = jt.full([B, W, H, H], jt.Var(float('-inf')))
        h_mask = h_mask.reindex(h_mask.shape, ['i0', 'i1', 'i2', 'i3'], overflow_conditions=[f'i3%{self.dilated}==i2%{self.dilated}'], overflow_value=0) # dilated
        w_mask = jt.full([B, H, W, W], jt.Var(float('-inf')))
        w_mask = w_mask.reindex(w_mask.shape, ['i0', 'i1', 'i2', 'i3'], overflow_conditions=[f'i3%{self.dilated}==i2%{self.dilated}'], overflow_value=0) # dilated


        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = jt.linalg.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(H) + h_mask
        energy_H = energy_H.transpose(1, 2)
        energy_W = jt.linalg.einsum('bchw,bchj->bhwj', query, key) + w_mask
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


class NeighborhoodCrissCrossAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def execute(self,x):
        B, C, H, W = x.shape
        quarter_H = H // 4
        quarter_W = W // 4

        # calculate dilated mask
        h_mask = jt.full([B, W, H, H], jt.Var(float('-inf')))
        h_mask = h_mask.reindex(h_mask.shape, ['i0', 'i1', 'i2', 'i3'], overflow_conditions=[f'(i3-i2)<{quarter_H}&&(i3-i2)>=0'], overflow_value=0) # neighborhood
        w_mask = jt.full([B, H, W, W], jt.Var(float('-inf')))
        w_mask = w_mask.reindex(w_mask.shape, ['i0', 'i1', 'i2', 'i3'], overflow_conditions=[f'(i3-i2)<<{quarter_W}&&(i3-i2)>=0'], overflow_value=0) # neighborhood

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = jt.linalg.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(H) + h_mask
        energy_H = energy_H.transpose(1, 2)
        energy_W = jt.linalg.einsum('bchw,bchj->bhwj', query, key) + w_mask
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
