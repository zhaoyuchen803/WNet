# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union
import monai
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep

class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=2)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class WNet(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        is_al: bool = False,
    ):

        super().__init__()
        self.is_al = is_al
        fea = ensure_tuple_rep(features, 6)
        print(f"features: {fea}.")

        self.conv_10 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_11 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_12 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_13 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_14 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_14 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_13 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_12 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_11 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.down_21 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        self.down_22 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        self.down_23 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        self.down_24 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_24 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_23 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_22 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_21 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        
        x0 = self.conv_10(x)

        x1 = self.down_11(x0)
        x2 = self.down_12(x1)
        x3 = self.down_13(x2)
        x4 = self.down_14(x3)

        u4 = self.upcat_14(x4, x3)
        u3 = self.upcat_13(u4, x2)
        u2 = self.upcat_12(u3, x1)
        u1 = self.upcat_11(u2, x0)

        if self.is_al:
            mid_logits = self.final_conv(u1)

        x1 = self.down_21(u1)
        x2 = self.down_22(x1)
        x3 = self.down_23(x2)
        x4 = self.down_24(x3)

        u4 = self.upcat_14(x4, x3)
        u3 = self.upcat_13(u4, x2)
        u2 = self.upcat_12(u3, x1)
        u1 = self.upcat_11(u2, u1)

        logits = self.final_conv(u1)

        if self.is_al:
            return logits, mid_logits

        return logits


if __name__ == "__main__":
    n_classes = 2
    net = WNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        is_al=True
    ).cuda()
    print(net)
    x = torch.ones((2,1,192,192,16)).cuda()
    x1, x2 = net(x)
    print(x1.shape, x2.shape)