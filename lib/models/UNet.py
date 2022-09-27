import torch
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv3d, Conv2d, MaxPool2d, MaxPool3d, InstanceNorm3d
from torch.nn import InstanceNorm2d, ReLU, AvgPool2d, AvgPool3d

class UNet(BaseModel):

    # Parameters of the model
    params = ["modalities", "n_classes", "dim"]
    def __init__(self, modalities, n_classes, dim="2D"):

        super(UNet, self).__init__()
        self.modalities = modalities
        self.n_classes = n_classes
        self.dim = dim

        if dim == "2D":
            Conv = Conv2d
            MP = MaxPool2d
        elif dim == "3D":
            Conv = Conv3d
            MP = MaxPool3d

        f = 1

        # Encoder
        self.enc_block11 = ConvBlock(modalities, 64/f, dim) # 64
        self.enc_block12 = ConvBlock(64/f, 64/f, dim) # 64
        self.mp1 = MP(2, ceil_mode=True)
        self.enc_block21 = ConvBlock(64/f, 128/f, dim) # 128
        self.enc_block22 = ConvBlock(128/f, 128/f, dim) # 128
        self.mp2 = MP(2, ceil_mode=True)
        self.enc_block31 = ConvBlock(128/f, 256/f, dim) # 256
        self.enc_block32 = ConvBlock(256/f, 256/f, dim) # 256
        self.mp3 = MP(2, ceil_mode=True)
        self.enc_block41 = ConvBlock(256/f, 512/f, dim) # 512
        self.enc_block42 = ConvBlock(512/f, 512/f, dim) # 512
        self.mp4 = MP(2, ceil_mode=True)

        self.enc_block51 = ConvBlock(512/f, 1024/f, dim) # 1024
        self.enc_block52 = ConvBlock(1024/f, 1024/f, dim) # 1024

        # Decoder
        self.upconv4 = UpConv(1024/f, 512/f, dim)
        self.dec_block41 = ConvBlock(512/f+512/f, 512/f, dim) # 512
        self.dec_block42 = ConvBlock(512/f, 512/f, dim) # 512
        self.upconv3 = UpConv(512/f, 256/f, dim)
        self.dec_block31 = ConvBlock(256/f+256/f, 256/f, dim) # 256
        self.dec_block32 = ConvBlock(256/f, 256/f, dim) # 256
        self.upconv2 = UpConv(256/f, 128/f, dim)
        self.dec_block21 = ConvBlock(128/f+128/f, 128/f, dim) # 128
        self.dec_block22 = ConvBlock(128/f, 128/f, dim) # 128
        self.upconv1 = UpConv(128/f, 64/f, dim)
        self.dec_block11 = ConvBlock(64/f+64/f, 64/f, dim) # 64
        self.dec_block12 = ConvBlock(64/f, 64/f, dim) # 64

        self.last = Conv(int(64/f), n_classes, 1)

    def forward(self, x):

        x = x[0]

        enc_block1_out = self.enc_block11(x)
        enc_block1_out = self.enc_block12(enc_block1_out)
        block1_size = enc_block1_out.size()[2:]
        x = self.mp1(enc_block1_out)

        enc_block2_out = self.enc_block21(x)
        enc_block2_out = self.enc_block22(enc_block2_out)
        block2_size = enc_block2_out.size()[2:]
        x = self.mp2(enc_block2_out)

        enc_block3_out = self.enc_block31(x)
        enc_block3_out = self.enc_block32(enc_block3_out)
        block3_size = enc_block3_out.size()[2:]
        x = self.mp3(enc_block3_out)

        enc_block4_out = self.enc_block41(x)
        enc_block4_out = self.enc_block42(enc_block4_out)
        block4_size = enc_block4_out.size()[2:]
        x = self.mp4(enc_block4_out)

        enc_block5_out = self.enc_block51(x)
        enc_block5_out = self.enc_block52(enc_block5_out)

        up4_out = self.upconv4(enc_block5_out, size=block4_size)
        x = torch.cat([enc_block4_out, up4_out], dim=1)
        dec_block4_out = self.dec_block41(x)
        dec_block4_out = self.dec_block42(dec_block4_out)

        up3_out = self.upconv3(dec_block4_out, size=block3_size)
        x = torch.cat([enc_block3_out, up3_out], dim=1)
        dec_block3_out = self.dec_block31(x)
        dec_block3_out = self.dec_block32(dec_block3_out)

        up2_out = self.upconv2(dec_block3_out, size=block2_size)
        x = torch.cat([enc_block2_out, up2_out], dim=1)
        dec_block2_out = self.dec_block21(x)
        dec_block2_out = self.dec_block22(dec_block2_out)

        up1_out = self.upconv1(dec_block2_out, size=block1_size)
        x = torch.cat([enc_block1_out, up1_out], dim=1)
        dec_block1_out = self.dec_block11(x)
        dec_block1_out = self.dec_block12(dec_block1_out)

        x = self.last(dec_block1_out)
        # Gather the distances between feature maps in all layers
        softed = torch.functional.F.softmax(x, dim=1)
        return (softed, )



class ConvBlock(torch.nn.Module):
    def __init__(self, fi_in, fi_out, dim):
        super(ConvBlock, self).__init__()

        fi_in, fi_out = int(fi_in), int(fi_out)

        if dim == "2D":
            Conv = Conv2d
        elif dim == "3D":
            Conv = Conv3d

        self.seq = torch.nn.Sequential(
                Conv(fi_in, fi_out, 3, padding=1),
                ReLU(),
                )

    def forward(self, x):
        return self.seq(x)

class UpConv(torch.nn.Module):
    def __init__(self, fi_in, fi_out, dim):
        super(UpConv, self).__init__()
        if dim == "2D":
            Conv = Conv2d
            self.interpol_mode = "bilinear"
        elif dim == "3D":
            Conv = Conv3d
            self.interpol_mode = "trilinear"

        fi_in, fi_out = int(fi_in), int(fi_out)

        self.conv = Conv(fi_in, fi_out, 3, padding=1)

    def forward(self, x, size):
        x = interpolate(x, size, mode=self.interpol_mode)
        x = self.conv(x)
        return x

