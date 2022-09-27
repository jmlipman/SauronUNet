from typing import Type, List, Dict
from lib.models.BaseModel import BaseModel, unwrap_data
import torch, os
import numpy as np
from torchio.data.dataset import SubjectsDataset
from torchio.data.subject import Subject
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
import nibabel as nib
from lib.models.Sauron import DropChannels
import pickle
import pandas as pd
from lib.utils import generate_merge_matrix_for_kernel, get_layer_idx_to_clusters, generate_decay_matrix_for_kernel_and_vecs, get_all_conv_kernel_namedvalue_as_dict, get_all_deps

"""
Callback functions are executed at particular times during the training or
validation of the models. The name of the callback function indicates when
it is executed. For now, all callback function's names must start with:
    * _start_epoch_: Executed at the beginning of each epoch.
    * _end_epoch_: Executed at the end of each epoch.
    * _start_train_iteration_: Executed at the beginning of each iteration.
    * _end_train_iteration_: Executed at the beginning of each iteration.
    * _start_val_subject: Executed at the beginning of each val sub iteration.
    * _end_val_subject: Executed at the end of each val sub iteration.

Callbacks' arguments must have the same name of the variables that expect.
These variables are gathered using "locals()", and passed to the callbacks.
Note that the model can be accessed with the parameter `self`.
"""


def _end_training_prune_csgd(self:Type[BaseModel], outputPath: str, e: int,
        channels_history: Dict[str, List[int]]):

    # PRUNE MODEL
    clusters = self.layer_idx_to_clusters
    # Part 1, prune the output filters
    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.imp_fun:
            keep_filters = []
            for clst in clusters[mod.name]:
                keep_filters.append(clst[0])

            with torch.no_grad():
                for inner_mod in mod.module.modules():
                    if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        inner_mod.weight = torch.nn.Parameter(inner_mod.weight[keep_filters])
                        inner_mod.bias = torch.nn.Parameter(inner_mod.bias[keep_filters])
                        inner_mod.out_channels = inner_mod.weight.shape[0]
                    elif isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                                torch.nn.ConvTranspose3d)):
                        inner_mod.weight = torch.nn.Parameter(inner_mod.weight[:, keep_filters])
                        inner_mod.bias = torch.nn.Parameter(inner_mod.bias[keep_filters])
                        inner_mod.out_channels = inner_mod.weight.shape[1]

    # Part 2, prune the input filters
    for mod in self.modules():
        if isinstance(mod, DropChannels) and len(mod.parents) > 0:
            #print(mod.name, keep_filters, mod.imp_fun)
            # Clusters from the input filters, from the parents
            # This accounts for the situation in which a conv block has
            # more than one parent, i.e., when concat is used.
            input_clusters = []
            offset = 0
            for parent in mod.parents:
                for cl in clusters[parent.name]:
                    cl = list(np.array(cl) + offset)
                    input_clusters.append(cl)
                offset += (np.hstack(clusters[parent.name]).max() + 1)

            with torch.no_grad():
                for inner_mod in mod.module.modules():
                    if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        keep_filters = []
                        for cl in input_clusters:
                            follow_kernel_value = inner_mod.weight.cpu().detach().numpy()
                            selected_k_follow = follow_kernel_value[:, cl]
                            summed_k_follow = np.sum(selected_k_follow, axis=1)
                            follow_kernel_value[:, cl[0]] = summed_k_follow
                            keep_filters.append(cl[0])
                        inner_mod.weight = torch.nn.Parameter(torch.as_tensor(follow_kernel_value[:, keep_filters]))
                        inner_mod.in_channels = inner_mod.weight.shape[1]

                        #from IPython import embed; embed()
                    elif isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                                torch.nn.ConvTranspose3d)):
                        keep_filters = []
                        for cl in input_clusters:
                            follow_kernel_value = inner_mod.weight.cpu().detach().numpy()
                            selected_k_follow = follow_kernel_value[cl]
                            summed_k_follow = np.sum(selected_k_follow, axis=0)
                            follow_kernel_value[cl[0]] = summed_k_follow
                            keep_filters.append(cl[0])
                        inner_mod.weight = torch.nn.Parameter(torch.as_tensor(follow_kernel_value[keep_filters]))
                        inner_mod.in_channels = inner_mod.weight.shape[0]

    # TRACK NUMBER OF FILTERS
    in_filters = {}
    for mod in self.modules():
        if not isinstance(mod, DropChannels):
            continue
        for submod in mod.modules():
            if isinstance(submod, (torch.nn.Conv2d, torch.nn.Conv3d,
                    torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                in_filters[mod.name] = submod.in_channels
                if not mod.name in channels_history:
                    channels_history[mod.name] = []
                # Before I used to save in_channels
                channels_history[mod.name].append(int(submod.out_channels))

    sorted_names = sorted(channels_history.keys())

    filePath_in = os.path.join(outputPath, "in_filters")
    if not os.path.isfile(filePath_in):
        with open(filePath_in, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    filePath_out = os.path.join(outputPath, "out_filters")
    if not os.path.isfile(filePath_out):
        with open(filePath_out, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    with open(filePath_out, "a") as f:
        f.write("\t".join([str(channels_history[n][-1]) for n in sorted_names]) + "\n")
    with open(filePath_in, "a") as f:
        f.write("\t".join([str(in_filters[n]) for n in sorted_names]) + "\n")

    # SAVED PRUNED MODEL
    path_models = os.path.join(outputPath, "models")
    e = e-1
    if "Sauron" in str(type(self)):
        # This is needed since Sauron introduces a few changes.
        # Without doing this, there will be problems when loading the weights
        state_dict_k = list(self.network.state_dict())
        state_dict = self.network.state_dict()
        for k in state_dict_k:
            newk = k.replace("module.", "")
            state_dict[newk] = state_dict[k]
            del state_dict[k]
        torch.save(state_dict,  f"{path_models}/model-{e}-pruned")
    else:
        torch.save(self.state_dict(),  f"{path_models}/model-{e}-pruned")
    #if e > 1 and os.path.exists(f"{path_models}/model-{e-1}"):
    #    os.remove(f"{path_models}/model-{e-1}")


def _after_compute_grads_csgd(self: Type[BaseModel], tr_loader: DataLoader):

    if tr_loader.dataset.dataset.dim == "2D":
        dims = 4
    elif tr_loader.dataset.dataset.dim == "3D":
        dims = 5


    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.name in self.merge_matrix:
            with torch.no_grad():
                for inner_mod in mod.module.modules():
                    if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        p_dim_w = inner_mod.weight.dim()
                        p_size_w = inner_mod.weight.size()
                        param_mat_w = inner_mod.weight.reshape(p_size_w[0], -1)
                        g_mat_w = inner_mod.weight.grad.reshape(p_size_w[0], -1)
                        csgd_gradient_w = self.merge_matrix[mod.name].matmul(g_mat_w) + self.decay_matrix[mod.name].matmul(param_mat_w)
                        inner_mod.weight.grad.copy_(csgd_gradient_w.reshape(p_size_w))

                    if isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                              torch.nn.ConvTranspose3d)):
                        # In Pytorch, the output filters in
                        # transposed convolutions are located in axis=1
                        p_dim_w = inner_mod.weight.dim()
                        p_size_w = inner_mod.weight.size()
                        param_mat_w = inner_mod.weight.reshape(p_size_w[1], -1)
                        g_mat_w = inner_mod.weight.grad.reshape(p_size_w[1], -1)
                        csgd_gradient_w = self.merge_matrix[mod.name].matmul(g_mat_w) + self.decay_matrix[mod.name].matmul(param_mat_w)
                        inner_mod.weight.grad.copy_(csgd_gradient_w.reshape(p_size_w))

                        # Bias is not done. see centripetal/utils/engine.py
                        # get_all_conv_kernel_namedvalue_as_list: it only
                        # gathers those with v.dim() == 4, i.e., the weights.

def _start_training_generate_matrices_csgd(self: Type[BaseModel],
        tr_loader: DataLoader, opt: Type[Optimizer], history: dict,
        outputPath: str):

    if "path" in history:
        # Load
        with open(os.path.join(history["path"], "layer_idx_to_clusters.pkl"), "rb") as f:
            self.layer_idx_to_clusters = pickle.load(f)
        with open(os.path.join(history["path"], "merge_matrix.pkl"), "rb") as f:
            self.merge_matrix = pickle.load(f)
        with open(os.path.join(history["path"], "decay_matrix.pkl"), "rb") as f:
            self.decay_matrix = pickle.load(f)
    else:
        num_classes = len(tr_loader.dataset.dataset.classes)
        if tr_loader.dataset.dataset.dim == "2D":
            dims = 4
        elif tr_loader.dataset.dataset.dim == "3D":
            dims = 5

        kernel_namedvalue_dict = get_all_conv_kernel_namedvalue_as_dict(self, dims)
        deps = get_all_deps(kernel_namedvalue_dict)

        # Target deps: We can either load output_filters directly or compute
        # the reduction rate
        reduction_rate = "/media/miguelv/HD1/Projects/Sauron/results/comparison/rats/Sauron/1/out_filters"
        reduction_rate = 0.29

        if isinstance(reduction_rate, str):
            # Load
            target_deps = dict(pd.read_csv(reduction_rate, sep="\t").iloc[-1])
        else:
            target_deps = {}
            for k in deps:
                if deps[k]==num_classes:
                    target_deps[k] = num_classes
                else:
                    target_deps[k] = int(deps[k]*(1-reduction_rate))

        self.layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_dict, target_deps)

        self.merge_matrix = generate_merge_matrix_for_kernel(deps,
                self.layer_idx_to_clusters, kernel_namedvalue_dict)

        weight_decay_bias = 0
        weight_decay = opt.state_dict()["param_groups"][0]["weight_decay"]
        # Originally, authors used: 3e-3 and 0.05
        # In rats, I found that 0.05 is too high and the training fails

        centri_strength = 1e-2

        self.decay_matrix = generate_decay_matrix_for_kernel_and_vecs(deps, self.layer_idx_to_clusters, kernel_namedvalue_dict, weight_decay, weight_decay_bias, centri_strength)

        # Save
        with open(os.path.join(outputPath, "layer_idx_to_clusters.pkl"), "wb") as f:
            pickle.dump(self.layer_idx_to_clusters, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(outputPath, "merge_matrix.pkl"), "wb") as f:
            pickle.dump(self.merge_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(outputPath, "decay_matrix.pkl"), "wb") as f:
            pickle.dump(self.decay_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


def _end_epoch_examine_clusters_csgd(self: Type[BaseModel], log, e):
    import itertools
    from scipy.spatial import distance

    # From the first conv
    #layers = [("enc_ConvBlock_1", 0), ("enc_ConvBlock_10", -1)]
    #for layer, idx in layers:
    min_epochs = 200 # This depends on the dataset
    eps = 1e-5 # If all layers have an avg distance lower than this, stop

    all_layer_names, all_avgs = [], []
    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.imp_fun:
            layer = mod.name
            all_layer_names.append(layer)
            clusters = self.layer_idx_to_clusters[layer]
            # Find the Conv layer and get its weights
            for inner_mod in mod.module.modules():
                if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    weights = inner_mod.weight.cpu().detach().numpy()
                    weights = np.reshape(weights, (weights.shape[0], -1))

                elif isinstance(inner_mod,
                        (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                    weights = inner_mod.weight.cpu().detach().numpy()
                    weights = np.reshape(weights, (weights.shape[1], -1))

            #from IPython import embed; embed()
            avg = []
            # For every cluster in a specific weight matrix, check the distance
            # between the difference filters that comprise the cluster.
            # This difference/distance is what centripetal SGD minimizes, so
            # that when deleting the filters, they are the same.
            for c in clusters:
                combs = list(itertools.combinations(c, 2))
                for com_1, com_2 in combs:
                    avg.append( distance.euclidean(weights[com_1], weights[com_2]) )
            all_avgs.append(np.mean(avg))
    all_avgs = np.array(all_avgs)

    # The avg distance in all layers is below that threshold, so we can
    # stop training and prune.
    if (all_avgs < eps).sum() == len(all_avgs) and e >= min_epochs:
        self.end_training = True

    max_idx = np.argmax(all_avgs)
    log(f"Average distances in the largest-distance layer `{all_layer_names[max_idx]}`: {all_avgs[max_idx]}")

def _end_epoch_save_all_FMs(self: Type[BaseModel],
        tr_loader: DataLoader, e: int, outputPath: str) -> None:

    #filePath_in = os.path.join(outputPath, "in_filters")
    fms_path = os.path.join(outputPath, "fms")
    if not os.path.isdir(fms_path):
        os.makedirs(fms_path)
    fms_path = os.path.join(fms_path, str(e))
    if not os.path.isdir(fms_path):
        os.makedirs(fms_path)

    for tr_i, subjects_batch in enumerate(tr_loader):
        break

    info = subjects_batch["info"]
    X, Y, W = unwrap_data(subjects_batch, tr_loader.dataset.dataset,
                          self.device)
    oneImage = [ X[0][0:1] ]
    FMs = self.forward_saveFMs(oneImage)

    for name, fm in FMs:
        np.save(os.path.join(fms_path, f"{name}"), fm)

def _end_epoch_save_history(self: Type[BaseModel], val_loss_history: List,
        tr_loss_history: List, channels_history: Dict[str, List[int]],
        outputPath: str) -> None:

    with open(os.path.join(outputPath, "val_loss_history.pkl"), "wb") as f:
        pickle.dump(val_loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "tr_loss_history.pkl"), "wb") as f:
        pickle.dump(tr_loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "channels_history.pkl"), "wb") as f:
        pickle.dump(channels_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    patiences, thrs = {}, {} # rho, tau
    for mod in self.modules():
        if isinstance(mod, DropChannels):
            patiences[mod.name] = mod.patience
            thrs[mod.name] = mod.thr

    with open(os.path.join(outputPath, "mod_patience.pkl"), "wb") as f:
        pickle.dump(patiences, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, "mod_thr.pkl"), "wb") as f:
        pickle.dump(thrs, f, protocol=pickle.HIGHEST_PROTOCOL)


def _end_epoch_prune(self: Type[BaseModel], log, val_loss_history: List,
        tr_loss_history: List, opt,
        channels_history: Dict[str, List[int]]) -> None:

    tau_max = 0.3
    kappa = 15
    rho = 5
    mu = 2/100

    def convergence(channel_history, val_loss_history, tr_loss_history):
        if len(val_loss_history) < rho+1:
            return False

        tr_loss_history = np.array(tr_loss_history)
        val_loss_history = np.array(val_loss_history)

        # The very last loss values
        last_tr_loss = tr_loss_history[-1]
        last_val_loss = val_loss_history[-1]

        # The previous N values to the last
        prev_tr_loss = tr_loss_history[-rho-1:-1]
        #prev_val_loss = val_loss_history[-conv_N-1:-1]

        # Training loss got better, so don't increase the pruning thr
        if last_tr_loss < prev_tr_loss.min():
            print("Uno")
            return False

        # Training loss got worse, so don't increase the pruning thr
        if last_tr_loss > prev_tr_loss.max():
            print("Dos")
            return False

        # Validation loss got better, so don't increase the pruning thr
        if last_val_loss < val_loss_history.min():
            print("Tres")
            return False

        # Actually, I think this is useless, but I'm not sure.
        # The idea is that if it has not decreased too much, return true
        thr = int(np.ceil(channel_history[-2]*mu))
        if (channel_history[-2] - channel_history[-1]) < thr:
            print("Cuatro")
            return True

        print("Cinco")
        return False

    ### PART 1. Prune the channels with distance smaller than thr.
    # new_params and old_params keep track of the old and new parameters
    # which is utilized below.
    new_params, old_params = [], []
    for mod in self.modules():
        if isinstance(mod, DropChannels) and mod.imp_fun:
            if mod.patience > 0:
                mod.patience -= 1
            elif (mod.name in channels_history
                    and convergence(channels_history[mod.name],
                                    val_loss_history,
                                    tr_loss_history) ): # convergencea
                # (rho) Default patience value to avoid moving the threshold
                mod.patience = rho
                # Update threshold linearly
                thrs = np.linspace(0.001, tau_max, kappa+1) # tau_max, kappa
                if mod.thr != thrs[-1]:
                    idx = mod.thr < thrs
                    mod.thr = thrs[idx][0]

                log(f"Increasing thr to {np.round(mod.thr, 3)} in {mod.name}")

            mod.remove_channels = mod.delta_prune < mod.thr
            # Ignore the channel used as a reference to compute the distances
            mod.remove_channels[mod.ref_idx] = False
            log(f"Deleting {mod.remove_channels.sum()} filters in {mod.name}; thr={mod.thr}")
            #raise Exception("add the channels_history thing")

            with torch.no_grad():
                for inner_mod in mod.module.modules():
                    # If shape[0] == 1, the script has skipped the code
                    # that produced new mod.avg_distance_across_batches
                    # and, as a consequence, the mod.avg... will be the
                    # same from the prev. iteration, and since
                    # it was pruned (that's why shape[0] == 1 now),
                    # it will raise an error as the shape is diff.
                    if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        #torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):

                        tmp_param_weight = inner_mod.weight
                        tmp_param_bias = inner_mod.bias
                        old_params.append(id(tmp_param_weight))
                        old_params.append(id(tmp_param_bias))

                        inner_mod.weight = torch.nn.Parameter(
                                tmp_param_weight[~mod.remove_channels])
                        inner_mod.bias = torch.nn.Parameter(
                                tmp_param_bias[~mod.remove_channels])
                        inner_mod.out_channels = torch.sum(
                                ~mod.remove_channels).cpu().detach().numpy()
                        new_params.append(inner_mod.weight)
                        new_params.append(inner_mod.bias)

                        # Keep step, exp_avg, and exp_avg_sq (for Adam
                        # optimizer). This depends on the optimizer.
                        # If SGD + momentum is utilized then you can
                        # replace the six lines below with these two:
                        # self.opt.state[inner_mod.weight]["momentum_buffer"] = self.opt.state[tmp_param_weight]["momentum_buffer"]
                        #self.opt.state[inner_mod.bias]["momentum_buffer"] = self.opt.state[tmp_param_bias]["momentum_buffer"]
                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.bias]["step"] = opt.state[tmp_param_bias]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][~mod.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg"] = opt.state[tmp_param_bias]["exp_avg"][~mod.remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][~mod.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg_sq"] = opt.state[tmp_param_bias]["exp_avg_sq"][~mod.remove_channels]

                        del opt.state[tmp_param_weight]
                        del opt.state[tmp_param_bias]
                        del tmp_param_weight
                        del tmp_param_bias
                    elif isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                                torch.nn.ConvTranspose3d)):

                        tmp_param_weight = inner_mod.weight
                        tmp_param_bias = inner_mod.bias
                        old_params.append(id(tmp_param_weight))
                        old_params.append(id(tmp_param_bias))

                        inner_mod.weight = torch.nn.Parameter(
                                tmp_param_weight[:, ~mod.remove_channels])
                        inner_mod.bias = torch.nn.Parameter(
                                tmp_param_bias[~mod.remove_channels])
                        inner_mod.out_channels = torch.sum(
                                ~mod.remove_channels).cpu().detach().numpy()
                        new_params.append(inner_mod.weight)
                        new_params.append(inner_mod.bias)


                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.bias]["step"] = opt.state[tmp_param_bias]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][:, ~mod.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg"] = opt.state[tmp_param_bias]["exp_avg"][~mod.remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][:, ~mod.remove_channels]
                        opt.state[inner_mod.bias]["exp_avg_sq"] = opt.state[tmp_param_bias]["exp_avg_sq"][~mod.remove_channels]

                        del opt.state[tmp_param_weight]
                        del opt.state[tmp_param_bias]
                        del tmp_param_weight
                        del tmp_param_bias


    # After the weights and their corresponding optimizer statistics are pruned
    # the optimizer needs to know which parameters are going to be optimized.
    # Such parameters are stored in ["params"].
    # You could tempted of doing this:
    #   self.opt.param_groups[0]["params"] = new_params
    #
    # However, new_params only contains the parameters from DropChannelWrapper
    # with mod.prune_mod = True. In other words, it does *not* contain other
    # parameters that you might have not wanted to prune (e.g., imagine
    # there is a specific layer that you didn't want to prune for some reason).
    # For this reason, this script saved old_params to not lose them.
    replace_params = []
    for p in opt.param_groups[0]["params"]:
        # Keeps old parameters that were not subject to pruning.
        if not id(p) in old_params:
            replace_params.append(p)
    # Add the rest of the parameters that were subject to pruning.
    replace_params += new_params
    opt.param_groups[0]["params"] = replace_params

    ### PART 2: Remove input channels
    # This makes the network compatible with the changes applied in Part 1.
    # The code is similar to Part 1 with the exception that bias need not
    # be pruned since it is not affected by the input channels.
    new_params, old_params = [], []
    for i, mod in enumerate(self.modules()):
        if isinstance(mod, DropChannels) and len(mod.parents) > 0:
            remove_channels = torch.cat([p.remove_channels for p in mod.parents])

            with torch.no_grad():
                for inner_mod in mod.module.modules():
                    if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        #torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                        #if inner_mod.weight.shape[1] != remove_channels.shape[0]:
                        #    print("Skipping", inner_mod.weight.shape, remove_channels.shape)
                        #    continue
                        tmp_param_weight = inner_mod.weight
                        old_params.append(id(tmp_param_weight))

                        #print(mod.avg_distance_across_batches)
                        inner_mod.weight = torch.nn.Parameter(tmp_param_weight[:, ~remove_channels])
                        inner_mod.in_channels = int(torch.sum(~remove_channels).cpu().detach().numpy())

                        new_params.append(inner_mod.weight)

                        # For SGD + Momentum
                        #self.opt.state[inner_mod.weight]["momentum_buffer"] = self.opt.state[tmp_param_weight]["momentum_buffer"]
                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][:, ~remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][:, ~remove_channels]

                        del opt.state[tmp_param_weight]
                        del tmp_param_weight
                    elif isinstance(inner_mod, (torch.nn.ConvTranspose2d,
                                                torch.nn.ConvTranspose3d)):
                        tmp_param_weight = inner_mod.weight
                        old_params.append(id(tmp_param_weight))

                        inner_mod.weight = torch.nn.Parameter(tmp_param_weight[~remove_channels])
                        inner_mod.in_channels = int(torch.sum(~remove_channels).cpu().detach().numpy())

                        new_params.append(inner_mod.weight)

                        opt.state[inner_mod.weight]["step"] = opt.state[tmp_param_weight]["step"]
                        opt.state[inner_mod.weight]["exp_avg"] = opt.state[tmp_param_weight]["exp_avg"][~remove_channels]
                        opt.state[inner_mod.weight]["exp_avg_sq"] = opt.state[tmp_param_weight]["exp_avg_sq"][~remove_channels]

                        del opt.state[tmp_param_weight]
                        del tmp_param_weight

    replace_params = []
    for p in opt.param_groups[0]["params"]:
        if not id(p) in old_params:
            replace_params.append(p)
    replace_params += new_params
    opt.param_groups[0]["params"] = replace_params



def _end_epoch_track_number_filters(self: Type[BaseModel], outputPath: str,
        channels_history: Dict[str, List[int]]):
    """
    Record the number of input filters in every conv. layer within
    'DropChannelWrapper' objects.
    Save all channels in `channel_history` to enable convergence detection.

    Args:
      `self`: model.
      `path_handler` (lib.utils.handlers.PathHandler).
      `channel_history`: Number of input chann. per DropChannelWrapper object.
    """
    """
    if (not hasattr(self, "dropchannelModules")
            or not isinstance(self.dropchannelModules, list)):
        message = ("The network must have an attribute named"
                "'dropchannelModule' that is of type 'list'."
                "It should contain the DropChannels modules")
        raise Exception(message)
    """

    in_filters = {}
    for mod in self.modules():
        if not isinstance(mod, DropChannels):
            continue
        for submod in mod.modules():
            if isinstance(submod, (torch.nn.Conv2d, torch.nn.Conv3d,
                    torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                in_filters[mod.name] = submod.in_channels
                if not mod.name in channels_history:
                    channels_history[mod.name] = []
                # Before I used to save in_channels
                channels_history[mod.name].append(int(submod.out_channels))

    sorted_names = sorted(channels_history.keys())

    filePath_in = os.path.join(outputPath, "in_filters")
    if not os.path.isfile(filePath_in):
        with open(filePath_in, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    filePath_out = os.path.join(outputPath, "out_filters")
    if not os.path.isfile(filePath_out):
        with open(filePath_out, "w") as f:
            f.write("\t".join([n for n in sorted_names]) + "\n")

    with open(filePath_out, "a") as f:
        f.write("\t".join([str(channels_history[n][-1]) for n in sorted_names]) + "\n")
    with open(filePath_in, "a") as f:
        f.write("\t".join([str(in_filters[n]) for n in sorted_names]) + "\n")




def _end_epoch_save_last_model(self: Type[BaseModel],
        outputPath: str, e: int) -> None:
    """
    Saves the current Pytorch model.

    Args:
      `self`: model.
      `outputPath`: Path to the output (e.g., exp_name/21
      `e`: Current epoch.
    """
    #path_models = path_handler.join("models")
    path_models = os.path.join(outputPath, "models")
    if "Sauron" in str(type(self)):
        # This is needed since Sauron introduces a few changes.
        # Without doing this, there will be problems when loading the weights
        state_dict_k = list(self.network.state_dict())
        state_dict = self.network.state_dict()
        for k in state_dict_k:
            newk = k.replace("module.", "")
            state_dict[newk] = state_dict[k]
            del state_dict[k]
        torch.save(state_dict,  f"{path_models}/model-{e}")
    else:
        torch.save(self.state_dict(),  f"{path_models}/model-{e}")
    if e > 1 and os.path.exists(f"{path_models}/model-{e-1}"):
        os.remove(f"{path_models}/model-{e-1}")

    #print("Saving the model")
    #from IPython import embed; embed()

def _start_train_iteration_save_patch(tr_loader: DataLoader,
        X: List[Tensor], Y: List[Tensor], info: dict):

    # Save every patch from the batch
    X = X[0].cpu().detach().numpy()
    Y = Y[0].cpu().detach().numpy()
    for i in range(X.shape[0]):
        path = f"/home/miguelv/pythonUEF/synch/debug_flips/{i}-2.nii.gz"
        nib.save(nib.Nifti1Image(np.moveaxis(X[i], 0, -1), np.eye(4)), path)
        break
    raise Exception("para")

def _end_val_subject_save_subject(data: SubjectsDataset,
        y_pred_cpu: np.ndarray, subject: Subject, path_scores: str) -> None:
    """
    Saves a subject image after one validation step.

    Args:
      `data`: Contains the function 'save'.
      `y_pred_cpu`: Prediction.
      `subject`: Subject the prediction belongs to. Contains the original path.
      `path_scores`: String that contains the output folder.
    """

    path_output = path_scores[:path_scores.find("scores")]
    #from IPython import embed; embed()
    #raise Exception("para")
    data.dataset.save(y_pred_cpu,
            os.path.join(path_output, "preds", f"{subject.info['id']}.nii.gz"),
            subject.info["path"])

