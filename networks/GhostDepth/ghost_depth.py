"""Ghost-Depth: a lightweight encoder-decoder network for monocular depth
estimation, reimplemented from

    W. Quan, J. Liu, C. Xu, C. Han, J. Song, S. Qi,
    "Ghost-Depth: A Lightweight Encoder-Decoder Network for Monocular
    Depth Estimation", CNIOT '23, doi:10.1145/3603781.3603861

The paper reports 2.71M parameters, REL 0.149 and delta1 0.797 on
NYU-Depth V2 at 228x304 input. It composes three published building
blocks, which are implemented here directly from their own papers:

  - GhostNet [Han et al., CVPR 2020] as the encoder backbone, with the
    classification layers removed (paper sec. 3.1.1).
  - Ghost convolution modules replacing the 3x3 convolutions in the
    decoder's upsampling path (paper sec. 3.1.2).
  - iAFF, the iterative attentional feature fusion module
    [Dai et al., WACV 2021], on the 40-channel skip connection only;
    every other skip connection is a plain addition (paper sec. 3.1.3,
    ablation Table 3).

Architecture, following the paper's Fig. 1 and sec. 3.1.2. The encoder
taps five scales; the deepest one enters the decoder through a channel-
reducing convolution, then four upsampling modules each double the
resolution, fuse the matching encoder skip, and apply Ghost-A (reduces
channel count) followed by Ghost-B (keeps channel count, refines):

    input          3   x H     x W
    encoder /2    16   x H/2   x W/2    -> skip (add)
    encoder /4    24   x H/4   x W/4    -> skip (add)
    encoder /8    40   x H/8   x W/8    -> skip (iAFF)
    encoder /16   80   x H/16  x W/16   -> skip (add)
    encoder /32  160   x H/32  x W/32   -> decoder input
    decoder      160 -> 80 -> 40 -> 24 -> 16, ending at H/2 x W/2
    depth head    16 -> 1      at H/2   x W/2

The H/2 x W/2 output matches the paper, which trains against 152x114
depth maps for 304x228 inputs (sec. 4.1). Callers that want depth at the
full input resolution should upsample the result themselves.

REPRODUCTION NOTES -- the paper states the architecture in prose, not as
a layer table, so a few details are inferred. They are called out here
because they matter if these numbers are ever quoted as "Ghost-Depth":

  1. Skip taps. The ablation (Table 3) enumerates skip-fusion points at
     16, 24, 40 and 80 channels, which fixes the /16 tap at 80 channels
     -- i.e. the first (stride-2) block of GhostNet's stage 4, before
     that stage widens to 112. See _TAP_AFTER_STAGE.
  2. Encoder tail (`keep_final_conv`). "We removed some layers related
     to image classification in GhostNet" is read here as dropping the
     pool + 1280-d conv + linear classifier while keeping the 160 -> 960
     ConvBnAct. That yields 2.79M parameters; dropping it too yields
     2.57M. The paper reports 2.71M, between the two, so neither reading
     is exact -- the default keeps it, as the closer of the two.
  3. Ghost-A / Ghost-B use a 3x3 primary convolution, since the paper
     says they "take the place of 3*3 conventional convolution layers".
     GhostNet's own internal Ghost modules use a 1x1 primary instead.
  4. The paper's iAFF equations (1) and (2) are used verbatim. The
     reference open-aff implementation additionally scales both terms by
     2; that factor is NOT applied here, to match the paper as written.
  5. Ghost-Depth was trained on NYU-Depth V2 with an ImageNet-pretrained
     GhostNet encoder. In this repo it is trained from scratch on SUN
     RGB-D, matching how RT-MonoDepth and FastDepth are trained here, so
     the comparison isolates architecture rather than pretraining. Its
     accuracy is therefore NOT expected to reproduce the paper's numbers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=4, min_value=None):
    """GhostNet's channel rounding helper (Han et al., CVPR 2020)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # never round down by more than 10%
        new_v += divisor
    return new_v


class GhostModule(nn.Module):
    """Ghost module [Han et al., CVPR 2020, Fig. 2b].

    A cheap stand-in for an ordinary convolution: a primary convolution
    produces oup/ratio "intrinsic" feature maps, then a depthwise
    convolution ("cheap linear operation") derives the remaining
    "ghost" maps from them, and the two halves are concatenated.
    """

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation block as used inside GhostNet bottlenecks."""

    def __init__(self, in_chs, se_ratio=0.25, divisor=4):
        super().__init__()
        reduced_chs = _make_divisible(in_chs * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.act1(self.conv_reduce(x_se))
        x_se = self.conv_expand(x_se)
        return x * torch.clamp(x_se + 3.0, 0.0, 6.0).div(6.0)  # hard-sigmoid gate


class GhostBottleneck(nn.Module):
    """GhostNet bottleneck: expanding Ghost module -> optional depthwise
    stride conv -> optional SE -> projecting Ghost module, plus shortcut.
    """

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.0):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride,
                                     (dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride,
                          (dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        return x + self.shortcut(residual)


# GhostNet 1.0x configuration [Han et al., CVPR 2020, Table 7]:
# each group is a list of [kernel, expansion, out_channels, se_ratio, stride].
_GHOSTNET_CFGS = [
    [[3, 16, 16, 0, 1]],                                                    # 0: /2,  16
    [[3, 48, 24, 0, 2]],                                                    # 1: /4,  24
    [[3, 72, 24, 0, 1]],                                                    # 2: /4,  24
    [[5, 72, 40, 0.25, 2]],                                                 # 3: /8,  40
    [[5, 120, 40, 0.25, 1]],                                                # 4: /8,  40
    [[3, 240, 80, 0, 2]],                                                   # 5: /16, 80
    [[3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1],
     [3, 480, 112, 0.25, 1], [3, 672, 112, 0.25, 1]],                       # 6: /16, 112
    [[5, 672, 160, 0.25, 2]],                                               # 7: /32, 160
    [[5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1],
     [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1]],                          # 8: /32, 160
]

# Which cfg group each skip is tapped after, and the deepest (decoder
# input) tap. Chosen so the skip channel counts are 16/24/40/80 exactly
# as enumerated by the paper's Table 3 ablation.
_TAP_AFTER_STAGE = (0, 2, 4, 5)      # -> 16 (/2), 24 (/4), 40 (/8), 80 (/16)
_TAP_CHANNELS = (16, 24, 40, 80)
_BOTTLENECK_CHANNELS = 160           # after cfg group 8, at /32


class GhostNetEncoder(nn.Module):
    """GhostNet with the classification head removed (paper sec. 3.1.1),
    exposing the five multi-scale feature maps the decoder consumes.

    `keep_final_conv` retains GhostNet's last feature layer, the
    ConvBnAct that widens 160 -> 960 channels, dropping only the pooling
    + 1280-d conv + linear classifier that are specific to ImageNet
    classification. See the reproduction note in the module docstring:
    keeping it puts the parameter count at 2.79M against the paper's
    reported 2.71M, dropping it gives 2.57M.
    """

    def __init__(self, width: float = 1.0, keep_final_conv: bool = True):
        super().__init__()
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        stages = []
        for cfg in _GHOSTNET_CFGS:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                out_ch = _make_divisible(c * width, 4)
                mid_ch = _make_divisible(exp_size * width, 4)
                layers.append(GhostBottleneck(input_channel, mid_ch, out_ch, k, s, se_ratio))
                input_channel = out_ch
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

        if keep_final_conv:
            final_channel = _make_divisible(960 * width, 4)
            self.final_conv = nn.Sequential(
                nn.Conv2d(input_channel, final_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(final_channel),
                nn.ReLU(inplace=True),
            )
            input_channel = final_channel
        else:
            self.final_conv = None

        self.tap_channels = tuple(_make_divisible(c * width, 4) for c in _TAP_CHANNELS)
        self.bottleneck_channels = input_channel

    def forward(self, x):
        """Returns (skips, bottleneck) where skips is a list ordered from
        the shallowest (/2) to the deepest (/16) tap.
        """
        x = self.act1(self.bn1(self.conv_stem(x)))
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in _TAP_AFTER_STAGE:
                skips.append(x)
        if self.final_conv is not None:
            x = self.final_conv(x)
        return skips, x


class MSCAM(nn.Module):
    """Multi-scale channel attention module [Dai et al., WACV 2021].

    Sums a pointwise "local" channel-attention branch that keeps spatial
    resolution with a "global" branch computed on globally pooled
    features, so channel attention is applied at two scales at once.
    """

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        inter_channels = max(1, int(channels // r))
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.sigmoid(self.local_att(x) + self.global_att(x))


class iAFF(nn.Module):
    """Iterative attentional feature fusion [Dai et al., WACV 2021], as
    used by Ghost-Depth on the 40-channel skip (paper sec. 3.1.3, Fig. 2).

    Two chained AFF stages, exactly the paper's equations:
        A = M(X + Y) * X + (1 - M(X + Y)) * Y     (Eq. 1)
        Z = M(A)     * X + (1 - M(A))     * Y     (Eq. 2)
    Note the second stage re-weights the ORIGINAL X and Y, using only the
    attention map derived from A -- it does not fuse A itself.
    """

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        self.mscam1 = MSCAM(channels, r)
        self.mscam2 = MSCAM(channels, r)

    def forward(self, x, y):
        w1 = self.mscam1(x + y)
        a = w1 * x + (1.0 - w1) * y
        w2 = self.mscam2(a)
        return w2 * x + (1.0 - w2) * y


class UpsampleBlock(nn.Module):
    """One decoder stage (paper sec. 3.1.2): bilinear x2 -> fuse the
    matching encoder skip -> Ghost-A (reduces channels) -> Ghost-B
    (keeps channels, refines).

    `fusion` is "iaff" only where the paper puts it (the 40-channel
    skip); everywhere else it is "add", the cheaper option the paper
    settled on in Table 4.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, fusion: str = "add"):
        super().__init__()
        if fusion not in ("add", "iaff"):
            raise ValueError(f"unknown fusion mode: {fusion}")
        if in_channels != skip_channels:
            raise ValueError(
                f"skip fusion needs matching channels, got in={in_channels} skip={skip_channels}"
            )
        self.fusion = fusion
        self.iaff = iAFF(skip_channels) if fusion == "iaff" else None
        # Ghost-A / Ghost-B stand in for two 3x3 convolutions, so their
        # primary convolution is 3x3 rather than the 1x1 GhostNet uses
        # inside its bottlenecks.
        self.ghost_a = GhostModule(in_channels, out_channels, kernel_size=3, relu=True)
        self.ghost_b = GhostModule(out_channels, out_channels, kernel_size=3, relu=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.iaff(x, skip) if self.fusion == "iaff" else x + skip
        return self.ghost_b(self.ghost_a(x))


class GhostDepth(nn.Module):
    """Full Ghost-Depth model.

    Outputs a single-channel map at half the input resolution. The raw
    head output is unbounded; scripts/train_ghostdepth_sunrgbd.py clamps
    it into the project's indoor metric range identically at train and
    inference time, matching how FastDepth is handled in this repo.
    """

    def __init__(self, width: float = 1.0, iaff_channels: int = 40, keep_final_conv: bool = True):
        super().__init__()
        self.encoder = GhostNetEncoder(width=width, keep_final_conv=keep_final_conv)
        skip_chs = list(self.encoder.tap_channels)          # [16, 24, 40, 80] at 1.0x

        # "passed through a convolution layer to reduce the number of
        # channels (for the subsequent feature fusion)" -- brings the /32
        # bottleneck down to the deepest skip's channel count.
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(self.encoder.bottleneck_channels, skip_chs[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(skip_chs[-1]),
            nn.ReLU(inplace=True),
        )

        # Four upsampling modules, walking the skips deepest -> shallowest.
        blocks = []
        for i in range(len(skip_chs) - 1, -1, -1):
            in_ch = skip_chs[i]
            out_ch = skip_chs[i - 1] if i > 0 else skip_chs[0]
            fusion = "iaff" if in_ch == iaff_channels else "add"
            blocks.append(UpsampleBlock(in_ch, skip_chs[i], out_ch, fusion=fusion))
        self.up_blocks = nn.ModuleList(blocks)

        self.depth_head = nn.Conv2d(skip_chs[0], 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips, bottleneck = self.encoder(x)
        out = self.channel_reduce(bottleneck)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            out = block(out, skip)
        return self.depth_head(out)
