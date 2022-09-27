import torch
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv3d, Conv2d, InstanceNorm2d, InstanceNorm3d
from torch.nn import LeakyReLU, AvgPool2d, AvgPool3d
from torch.nn import ConvTranspose2d, ConvTranspose3d
import numpy as np
import os

# Details from the original nnUNet implementation and optimization strategy:
# Architecture:
# [x] No ResNet, DenseNet, Attention, SENet, or dilated convs.
# [x] Instance norm
# [x] LeakyRelu (negative slope = 0.01)
# [-] Deep supervision on the first three levels.
# [x] Strided convolutions to downsample
# [x] Transposed convolutions to upsample
# [x] Init feature maps: 32
# [x] Max feature maps: 320 (3D), 512 (2D)
# Optimization strategy:
# [-] 1000 epochs
# [-] 250 minibatches (how to do this? it depends on the dataset size, typically)
# [-] SGD with Nesterov momemtum \mu=0.99, lr=0.01
# [-] 'Poly' lr decay: (1 − epoch/epoch_max)**0.9
# [-] CEDiceLoss
# [-] Deep supervision with weights: 4/7 + 2/7 + 1/7
# [-] Oversampling for class imbalance (???)
# Data augmentation
# [-] Rotations
# [-] Scaling
# [-] Gaussian noise
# [-] Gaussian blur
# [-] Brightness
# [-] Contrast
# [-] Simulation of low resolution
# [-] Gamma correction
# [-] Mirroring


class nnUNet(BaseModel):
    # Actual differences wrt nnUNet
    # - Maxpooling instead of strided convs
    # - No maximum of feature maps: "To limit the final model size,
    #    the number of feature maps is additionally capped at 320 and 512
    #    for 3D and 2D U-Nets, respectively."

    # Parameters of the model
    params = ["modalities", "n_classes", "dim"]
    def __init__(self, modalities, n_classes, fms_init=48, levels=5,
            normLayer=InstanceNorm2d, fms_max=480, dim="2D", filters={}):

        # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        # weights = weights / weights.sum()

        super(nnUNet, self).__init__()
        self.modalities = modalities
        self.n_classes = n_classes
        self.fms_init = fms_init
        self.levels = levels
        self.normLayer = normLayer
        self.dim = dim

        if dim == "2D":
            Conv = Conv2d
            Transposed = ConvTranspose2d
        elif dim == "3D":
            Conv = Conv3d
            Transposed = ConvTranspose3d

        # Determine the number of input and output channels in each conv
        if len(filters) == 0:
            filters["in"], filters["out"] = {}, {}

            filters["in"]["enc_ConvBlock_1"] = modalities
            filters["out"]["enc_ConvBlock_1"] = fms_init
            filters["in"]["enc_ConvBlock_2"] = fms_init
            filters["out"]["enc_ConvBlock_2"] = fms_init

            for i in range(1, levels):
                filters["in"][f"enc_ConvBlock_{i*2+1}"] = filters["out"][f"enc_ConvBlock_{(i-1)*2+1}"]
                fs = np.clip(filters["in"][f"enc_ConvBlock_{i*2}"]*2, 0, fms_max)
                filters["out"][f"enc_ConvBlock_{i*2+1}"] = fs
                filters["in"][f"enc_ConvBlock_{i*2+2}"] = fs
                filters["out"][f"enc_ConvBlock_{i*2+2}"] = fs

            for i in range(levels-1):
                filters["in"][f"dec_ConvTranspose{dim.lower()}_{i+1}"] = filters["out"][f"enc_ConvBlock_{(levels-1-i)*2+1}"]
                filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"] = filters["in"][f"enc_ConvBlock_{(levels-1-i)*2+1}"]
                filters["in"][f"dec_ConvBlock_{i*2+1}"] = filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"]*2
                filters["out"][f"dec_ConvBlock_{i*2+1}"] = filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"]
                filters["in"][f"dec_ConvBlock_{i*2+2}"] = filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"]
                filters["out"][f"dec_ConvBlock_{i*2+2}"] = filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"]

            for i in range(2, levels):
                filters["in"][f"dec_Sequential_{i-1}"] = filters["out"][f"dec_ConvBlock_{i*2}"]
                filters["out"][f"dec_Sequential_{i-1}"] = n_classes

        # Encoder
        self.encoder = [ConvBlock(filters["in"]["enc_ConvBlock_1"],
                                  filters["out"]["enc_ConvBlock_1"], dim),
                        ConvBlock(filters["in"]["enc_ConvBlock_2"],
                                  filters["out"]["enc_ConvBlock_2"], dim)]


        #fs = [fms_init] # filters
        for i in range(1, levels):
            #fs.append ( np.clip(fs[-1]*2, 0, fms_max) )
            #self.encoder.append( ConvBlock(fs[-2], fs[-1], dim, strides=2) )
            #self.encoder.append( ConvBlock(fs[-1], fs[-1], dim) )
            self.encoder.append( ConvBlock(filters["in"][f"enc_ConvBlock_{i*2+1}"],
                         filters["out"][f"enc_ConvBlock_{i*2+1}"], dim, strides=2) )
            self.encoder.append( ConvBlock(filters["in"][f"enc_ConvBlock_{i*2+2}"],
                         filters["out"][f"enc_ConvBlock_{i*2+2}"], dim) )
        self.encoder = torch.nn.ModuleList(self.encoder)

        # Decoder
        self.decoder = []
        #fs = fs[::-1]
        for i in range(levels-1):
            # self.decoder.append( Transposed(fs[i], fs[i+1], 2,
            #                            stride=2) )
            # I think that it originally has bias=False
                                       #stride=2, bias=False) )
            #self.decoder.append( ConvBlock(fs[i+1]*2, fs[i+1], dim) )
            #self.decoder.append( ConvBlock(fs[i+1], fs[i+1], dim) )
            self.decoder.append( Transposed(
                filters["in"][f"dec_ConvTranspose{dim.lower()}_{i+1}"],
                filters["out"][f"dec_ConvTranspose{dim.lower()}_{i+1}"],
                2, stride=2) )
            self.decoder.append( ConvBlock(
                filters["in"][f"dec_ConvBlock_{i*2+1}"],
                filters["out"][f"dec_ConvBlock_{i*2+1}"], dim) )
            self.decoder.append( ConvBlock(
                filters["in"][f"dec_ConvBlock_{i*2+2}"],
                filters["out"][f"dec_ConvBlock_{i*2+2}"], dim) )
        self.decoder = torch.nn.ModuleList(self.decoder)

        # Output layers (deep supervision)
        # Starting from the deeper
        self.last = []
        for i in range(2, levels):
            # I think that it originally has bias=False
            #self.last.append( torch.nn.Sequential( Conv(fs[i], n_classes, 1, bias=False) ))
            self.last.append( torch.nn.Sequential( Conv(
                filters["in"][f"dec_Sequential_{i-1}"],
                filters["out"][f"dec_Sequential_{i-1}"], 1) )
                )
        self.last = torch.nn.ModuleList(self.last)

    def forward(self, x):

        x = x[0]

        # Encoder
        #print(len(self.encoder), len(self.decoder), len(self.last))
        skip_outputs = [] # len(skip_outputs) = levels
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            x = self.encoder[i+1](x)
            #print(i, x.shape)
            skip_outputs.append( x )

        #print("----")
        # Decoder + skip connections
        x = skip_outputs[-1] # Right before the tranposed convolution
        last_convs = [] # FMs prior to the last convolution at each level
        for i in range(0, len(self.decoder), 3):
            skip_idx = ((i//3)+2)*-1
            x = self.decoder[i](x)
            #print(i, x.shape, skip_outputs[skip_idx].shape)
            x = torch.cat([skip_outputs[skip_idx], x], dim=1)
            x = self.decoder[i+1](x)
            x = self.decoder[i+2](x)
            last_convs.append( x )

        # Last convs, outputs for deep supervision
        outputs = []
        for i in range(len(self.last)):
            outputs.append( torch.functional.F.softmax(
                    self.last[-i-1](last_convs[-i-1]), dim=1) )
            #print(i, outputs[-1].shape)

        return outputs

    def forward_debug(self, x):
        x = x[0]
        return self.encoder[0](x)

    def forward_saveFMs(self, x):

        #from IPython import embed; embed()
        #raise Exception("para")
        x = x[0]
        allFMs = []

        # Encoder
        skip_outputs = [] # len(skip_outputs) = levels
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            allFMs.append((self.encoder[i].name, x.cpu().detach().numpy()))
            x = self.encoder[i+1](x)
            allFMs.append((self.encoder[i+1].name, x.cpu().detach().numpy()))
            #print(i, x.shape)
            skip_outputs.append( x )

        #print("----")
        # Decoder + skip connections
        x = skip_outputs[-1] # Right before the tranposed convolution
        last_convs = [] # FMs prior to the last convolution at each level
        for i in range(0, len(self.decoder), 3):
            skip_idx = ((i//3)+2)*-1
            x = self.decoder[i](x)
            allFMs.append((self.decoder[i].name, x.cpu().detach().numpy()))
            #print(i, x.shape, skip_outputs[skip_idx].shape)
            x = torch.cat([skip_outputs[skip_idx], x], dim=1)
            x = self.decoder[i+1](x)
            allFMs.append((self.decoder[i+1].name, x.cpu().detach().numpy()))
            x = self.decoder[i+2](x)
            allFMs.append((self.decoder[i+2].name, x.cpu().detach().numpy()))
            last_convs.append( x )

        # Last convs, outputs for deep supervision
        outputs = []
        for i in range(len(self.last)):
            t = self.last[-i-1](last_convs[-i-1])
            outputs.append( torch.functional.F.softmax(
                    t, dim=1) )
            allFMs.append((self.last[-i-1].name, t.cpu().detach().numpy()))
            #print(i, outputs[-1].shape)

        return allFMs



class ConvBlock(torch.nn.Module):
    def __init__(self, fi_in, fi_out, dim, strides=1):
        super(ConvBlock, self).__init__()

        fi_in, fi_out = int(fi_in), int(fi_out)

        if dim == "2D":
            Conv = Conv2d
            Norm = InstanceNorm2d
        elif dim == "3D":
            Conv = Conv3d
            Norm = InstanceNorm3d

        self.seq = torch.nn.Sequential(
                Conv(fi_in, fi_out, 3, padding=1, stride=strides),
                Norm(fi_out),
                LeakyReLU(),
                )

    def forward(self, x):
        return self.seq(x)

