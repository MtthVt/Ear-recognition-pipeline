""" Full assembly of the parts to form the complete network """
from .utils import UnetGridGatingSignal2
from ...unet.unet_parts import *
from ..layers.grid_attention_layer import GridAttentionBlock2D


class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.gating = UnetGridGatingSignal2(filters[4] // factor, filters[3], kernel_size=(1, 1), is_batchnorm=True)

        nonlocal_mode='concatenation'
        attention_dsample = (2, 2)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Gating Signal Generation
        center = x5
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(x4, gating)
        g_conv3, att3 = self.attentionblock3(x3, gating)
        g_conv2, att2 = self.attentionblock2(x2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up1(center, g_conv4)
        up3 = self.up2(up4, g_conv3)
        up2 = self.up3(up3, g_conv2)
        up1 = self.up4(up2, x1)
        logits = self.outc(up1)

        return logits
