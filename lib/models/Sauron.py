import torch
from lib.models.BaseModel import BaseModel
from torch.nn.functional import interpolate
from torch.nn import Conv3d, Conv2d, InstanceNorm2d, InstanceNorm3d
from torch.nn import LeakyReLU, AvgPool2d, AvgPool3d
from torch.nn import ConvTranspose2d, ConvTranspose3d
import numpy as np
import os
from lib.models.nnUNet import nnUNet
import lib.distance as distance
import pandas as pd

distances = {
        "euc_norm": distance.Euclidean_norm,
        "euc_norm_nonorm": distance.Euclidean_norm_nonorm,
        "euc_norm_deltaprunenorm": distance.Euclidean_norm_deltaprunenorm,
        "euc_rand": distance.Euclidean_rand,
        "": None,
        }

class Sauron(BaseModel):
    """
    Sauron wraps an existing neural network.
    Inside Sauron's initialization, we wrap blocks with DropChannel.
    """

    # Parameters of the model
    params = ["modalities", "n_classes", "dim"]
    def __init__(self, modalities, n_classes,
            dist_fun: str="", imp_fun: str="", sf: int=2,
            fms_init=48, levels=5,
            normLayer=InstanceNorm2d, fms_max=480, dim="2D",
            filters={}):
        super(Sauron, self).__init__()
        # We save these properties (mandatory for logging)
        self.modalities = modalities
        self.n_classes = n_classes
        self.dim = dim

        """
        path_history = ""
        path_history = "/home/miguelv/data/out/RAW/Sauron_reimp/implementing/rats/baseline/26/"
        if path_history == "":
            filters = {}
        else:
            df_in = pd.read_csv(os.path.join(path_history, "in_filters"), sep="\t")
            df_out = pd.read_csv(os.path.join(path_history, "out_filters"), sep="\t")
            filters = {}
            filters["in"] = {col_name: df_in[col_name].iloc[-1] for col_name in df_in.columns}
            filters["out"] = {col_name: df_out[col_name].iloc[-1] for col_name in df_out.columns}
        """

        # Initialize the CNN
        self.network = nnUNet(modalities, n_classes, fms_init, levels,
                normLayer, fms_max, dim, filters=filters)

        # Wrap each "block" with DropChannels. The following code will go
        # through different blocks within nnUNet and will wrap them.
        # Important information to wrap each code:
        # - name: name of a block, for debugging reasons.
        # - parents: list of blocks that will provide the input
        # - dist_fun: distance that will be minimized (delta_opt)
        # - imp_fun: distance that will be used for thresholding (delta_prune)

        dist_fun = distances[dist_fun]
        imp_fun = distances[imp_fun]
        if dim == "3D":
            compress = torch.nn.AvgPool3d(sf)
        elif dim == "2D":
            compress = torch.nn.AvgPool2d(sf)

        # Encoder
        tmp_d = {}
        for i in range(len(self.network.encoder)):
            name_module = self.network.encoder[i]._get_name()
            tmp_d[name_module] = tmp_d.get(name_module, 0)+1
            name = f"enc_{name_module}_{tmp_d[name_module]}"
            parents = [self.network.encoder[i-1]] if i>0 else []

            params = { "module": self.network.encoder[i],
                       "name": name, "parents": parents,
                       "dist_fun": dist_fun, "imp_fun": imp_fun,
                       "compress": compress}
            self.network.encoder[i] = DropChannels(**params)

        # Decoder
        tmp_d = {}
        for i in range(len(self.network.decoder)):
            name_module = self.network.decoder[i]._get_name()
            tmp_d[name_module] = tmp_d.get(name_module, 0)+1
            name = f"dec_{name_module}_{tmp_d[name_module]}"

            if "ConvTranspose" in name_module:
                if i>0:
                    parents = [self.network.decoder[i-1]]
                else:
                    parents = [self.network.encoder[-1]]
            elif "ConvBlock" in name_module:
                parents = [self.network.decoder[i-1]]
                if tmp_d[name_module] % 2 == 1:
                    # Handling the Skip connection
                    idx_skip = -2-tmp_d[name_module]
                    parents = [self.network.encoder[idx_skip], parents[0]]

            params = { "module": self.network.decoder[i],
                       "name": name, "parents": parents,
                       "dist_fun": dist_fun, "imp_fun": imp_fun,
                       "compress": compress}
            self.network.decoder[i] = DropChannels(**params)

        # Last layers (part of Deep Supervision)
        levels = len(self.network.decoder)//3 - 1
        for i in range(len(self.network.last)):
            name_module = self.network.last[i]._get_name()
            tmp_d[name_module] = tmp_d.get(name_module, 0)+1
            name = f"dec_{name_module}_{tmp_d[name_module]}"

            idx_dec = (-levels+i)*3+2
            parents = [self.network.decoder[idx_dec]]

            params = { "module": self.network.last[i],
                       "name": name, "parents": parents,
                       "dist_fun": None, "imp_fun": None, # (*)
                       "compress": compress}
            # (*): The output FMs of these convs should not be pruned
            #      because fi_out = n_classes
            self.network.last[i] = DropChannels(**params)

        # Important! This way we know where to find those modules
        # Used in _end_epoch_track_number_of_filters callback
        #self.dropchannelModules = [self.network.encoder,
        #        self.network.decoder, self.network.last]

    def forward(self, x):

        outputs = self.network(x)

        # Gather here the distances
        distances = []
        for arch_part in [self.network.encoder, self.network.decoder,
                          self.network.last]:
            for mod in arch_part:
                if not isinstance(mod, DropChannels):
                    message = f"Module '{mod.name}' should be `DropChannels`"
                    raise ValueError(message)
                distances.append(mod.delta_opt)

        #from IPython import embed; embed()
        #raise Exception("para")

        return (outputs, distances)

    def forward_saveFMs(self, x):

        FMs = self.network.forward_saveFMs(x)
        return FMs


class DropChannels(torch.nn.Module):
    """
    DropChannels must wrap a module that contains a convolution or a sequence,
    typically containing convolution+norm+act.
    """
    thr = 0.001 # Initial thresholds
    patience = 0 # To avoid pruning too fast
    def __init__(self, module, name: str, parents: list,
            dist_fun, imp_fun, compress):
        super(DropChannels, self).__init__()
        self.name = name
        self.module = module
        self.parents = parents
        self.dist_fun = dist_fun
        self.imp_fun = imp_fun
        self.compress = compress

    def forward(self, x):
        out = self.module(x)

        self.delta_opt = torch.ones((out.shape[0], 1)).cuda()
        self.delta_prune = torch.ones((1)).cuda()
        self.ref_idx = 0

        # If it's 1, there is nothing to prune
        if self.dist_fun and out.shape[1]:
            self.delta_opt = self.dist_fun(out, self.compress)

        if self.imp_fun and out.shape[1] > 1:
            self.delta_prune, self.ref_idx = self.imp_fun(out, self.compress)

        #print(self.name)
        #print(self.delta_opt)
        #print(self.delta_prune)
        #print(self.ref_idx)

        return out

