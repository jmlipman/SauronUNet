import argparse, inspect, os, sys
from lib.data.BaseDataset import BaseDataset
from typing import Type
from torch.nn.parameter import Parameter as TorchParameter
import torch, json
from lib.paths import data_path
import numpy as np
import types, random, time, pickle
from datetime import datetime
from torch import Tensor
import torchio as tio
from torch.utils.data import DataLoader
import nibabel as nib
from typing import List, Tuple, Union
import pandas as pd
from sklearn.cluster import KMeans

def getPCname() -> str:
    pc_name = os.uname()[1]
    if "bullx" in pc_name:
        pc_name = "CSC"
    if "sampo" in pc_name:
        pc_name = "sampo"
    return pc_name


def getDataset(dataset: str) -> Type[BaseDataset]:
    """
    Retrieves the dataset given its Dataset.name (files in lib/data/)

    Args:
      `dataset`: name of the dataset.

    Returns:
      Dataset object.
    """
    # Datasets
    # Placing the imports here avoid circular import
    if dataset == "rats":
        from lib.data.RatsDataset import RatsDataset
        return RatsDataset
    elif dataset == "acdc17":
        from lib.data.ACDC17Dataset import ACDC17Dataset
        return ACDC17Dataset
    elif dataset == "kits19":
        from lib.data.KiTS19Dataset import KiTS19Dataset
        return KiTS19Dataset

    raise ValueError(f"Dataset `{dataset}` not found.")


def parseArguments() -> None:
    """
    Parses, verifies, and sanitizes the arguments provided by the user.
    """

    parser = argparse.ArgumentParser(description="UNet strikes back! parser")

    parser.add_argument("--exp_name", help="Name of the experiment",
            default="baseline")

    # Data and model
    parser.add_argument("--data", help="Name of the dataset", required=True)
    parser.add_argument("--split", help="Train-Validation splits", default="1:0")
    parser.add_argument("--device", help="Pytorch device", default="cuda")
    parser.add_argument("--model_state", help="Pretrained model", default="")
    parser.add_argument("--in_filters", help="File containing in_filters", default="")
    parser.add_argument("--out_filters", help="File containing out_filters", default="")

    # Training strategy
    parser.add_argument("--loss", help="Name of the loss function",
            default="")
    # epochs_start can be useful to continue the training when using lr_decay
    parser.add_argument("--epoch_start", help="Number of epochs", default=1)
    parser.add_argument("--epochs", help="Number of epochs", default="")
    parser.add_argument("--batch_size", help="Batch size", default="")
    parser.add_argument("--val_interval", help=f"Frequency in which validation"
            " will be computed", default=2)
    parser.add_argument("--optim", help="Optimizer (e.g., adam)", default="")
    parser.add_argument("--lr", help="Learning rate", default="")
    parser.add_argument("--wd", help="Weight decay", default="")
    parser.add_argument("--momentum", help="Momentum (use with SGD)", default="")
    parser.add_argument("--nesterov", help="Momentum (use with SGD)", default="")

    # Other
    parser.add_argument("--seed_train", help="Random seed for pytorch, np, random",
            default="")
    parser.add_argument("--seed_data", help="Random seed for partitioning the data",
            default="")
    parser.add_argument("--history", help="Location of val_loss_history, etc. Useful for loading Sauron's state.", default="")

    args = parser.parse_args()

    cfg = {"data": getDataset(args.data)}

    # OPTIMIZER
    # If the user specifies the optimizer, it should also specify other
    # optimizer-related params, such as lr.
    optim_name = args.optim.lower()
    # Use default config. specific to the dataset
    if optim_name == "":
        cfg["optim"] = cfg["data"].opt["optim"]
    elif optim_name  == "adam":
        cfg["optim"] = torch.optim.Adam
    elif optim_name  == "sgd":
        cfg["optim"] = torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer `{args.optim}`"
                f" only 'adam' and 'sgd' available at the moment")

    # Grab the default opt params, and override those provided by the user
    cfg["optim_params"] = cfg["data"].opt["optim_params"]

    if args.lr != "":
        # If lr is not a number -> ValueError
        cfg["optim_params"]["lr"] = float(args.lr)
    if args.wd != "":
        cfg["optim_params"]["weight_decay"] = float(args.wd)

    if cfg["optim"] is torch.optim.SGD:
        if args.momentum != "":
            cfg["optim_params"]["momentum"] = float(args.momentum)
        if args.nesterov != "":
            cfg["optim_params"]["nesterov"] = bool(args.nesterov)

    # LOSS FUNCTION
    if args.loss == "":
        cfg["loss"] = cfg["data"].opt["loss"]
    else:
        loss_name = args.loss.lower()
        tt = __import__("lib.losses")
        allowed_losses = []
        for name, obj in inspect.getmembers(tt.losses, inspect.isfunction):
            if name.lower() == loss_name:
                cfg["loss"] = obj
            allowed_losses.append(name)
        if "loss" not in cfg:
            raise ValueError(f"Unknown loss function `{args.loss}`"
                    f" choose one from lib.losses.py: {allowed_losses}")

    # SCHEDULER (LR DECAY)
    if "scheduler" in cfg["data"].opt:
        cfg["scheduler"] = cfg["data"].opt["scheduler"]
        cfg["scheduler_params"] = cfg["data"].opt["scheduler_params"]

    # OTHER
    pc_name = getPCname()
    cfg["data"].data_path = data_path[args.data][pc_name]
    cfg["architecture"] = cfg["data"].opt["architecture"]
    if args.epochs != "":
        cfg["epochs"] = int(args.epochs)
    else:
        cfg["epochs"] = cfg["data"].opt["epochs"]

    if args.batch_size != "":
        cfg["batch_size"] = int(args.epochs)
    else:
        cfg["batch_size"] = cfg["data"].opt["batch_size"]

    cfg["val_interval"] = int(args.val_interval)
    cfg["epoch_start"] = int(args.epoch_start)

    # This "history" is only used with Sauron
    cfg["history"] = {}
    if args.history != "" and os.path.isdir(args.history):
        cfg["history"]["path"] = args.history
        with open(os.path.join(args.history, "val_loss_history.pkl"), "rb") as f:
            cfg["history"]["val_loss_history"] = pickle.load(f)
        with open(os.path.join(args.history, "tr_loss_history.pkl"), "rb") as f:
            cfg["history"]["tr_loss_history"] = pickle.load(f)
        with open(os.path.join(args.history, "channels_history.pkl"), "rb") as f:
            cfg["history"]["channels_history"] = pickle.load(f)
        with open(os.path.join(args.history, "mod_thr.pkl"), "rb") as f:
            cfg["history"]["mod_thr"] = pickle.load(f)
        with open(os.path.join(args.history, "mod_patience.pkl"), "rb") as f:
            cfg["history"]["mod_patience"] = pickle.load(f)

        cfg["architecture"]["filters"] = {}
        df_in = pd.read_csv(os.path.join(args.history, "in_filters"), sep="\t")
        df_out = pd.read_csv(os.path.join(args.history, "out_filters"), sep="\t")
        cfg["architecture"]["filters"]["in"] = {col_name: df_in[col_name].iloc[-1] for col_name in df_in.columns}
        cfg["architecture"]["filters"]["out"] = {col_name: df_out[col_name].iloc[-1] for col_name in df_out.columns}
    else:
        cfg["history"]["channels_history"] = {}
        cfg["history"]["val_loss_history"] = []
        cfg["history"]["tr_loss_history"] = []
        cfg["architecture"]["filters"] = {}

    # For some reason, I need to import this here. Otherwise I would need to
    # use 'global', which I will avoid at all costs
    from lib.paths import output_path
    if os.path.isdir(output_path[pc_name]):
        cfg["path"] = output_path[pc_name]
    else:
        raise ValueError(f"Output path `{output_path[pc_name]}` set in 'lib/paths.py'"
                f" is not a folder")

    if args.model_state != "":
        if not os.path.isfile(args.model_state):
            raise ValueError(f"The pretrained model specified in --model_state"
                    f" `{args.model_state}` does not exist.")
    cfg["model_state"] = args.model_state

    if args.device in ["cuda", "cpu"]:
        cfg["device"] = args.device
    else:
        raise ValueError(f"Unknown device `{args.device}`."
                          " Valid options: cuda, cpu")

    split = args.split.split(":")
    if len(split) != 2:
        raise ValueError("--split should contain two values separated by :"
                ", e.g., 0.9:0:1, specifying the train-validation splits")
    split = [float(s) for s in split if 0 <= float(s) <= 1]
    if sum(split) != 1 or len(split) != 2:
        raise ValueError("---split values should sum up to 1 and each value"
                " must within the range [0,1]")
    cfg["split"] = split

    if args.seed_train == "":
        seed_train = int(str(int(np.random.random()*1000)) + str(time.time()).split(".")[-1][:4])
    else:
        seed_train = int(args.seed_train)

    if args.seed_data == "":
        seed_data = int(str(int(np.random.random()*1000)) + str(time.time()).split(".")[-1][:4])
    else:
        seed_data = int(args.seed_data)

    torch.manual_seed(seed_train)
    torch.cuda.manual_seed(seed_train)
    np.random.seed(seed_train)
    random.seed(seed_train)
    cfg["seed_train"] = seed_train
    cfg["seed_data"] = seed_data

    return cfg


class Log:
    """
    Prints and stores the output of the experiments.
    """
    def __init__(self, path: str):
        """
        Args:
          `path`: Log file path.
        """
        self.path = path
        #self.log_path = path_handler.join("log.txt")
        #self.config_path = path_handler.join("config.json")

    def __call__(self, text: str, verbose: bool=True):
        """
        Saves the text into a log file.

        Args:
          `text`: Text to log.
          `verbose`: Whether to print `text` (0/1).
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{now}: {text}"
        if self.path != "": # Empty path -> disables log
            with open(self.path, "a") as f:
                f.write(text + "\n")

        if verbose:
            print(text)

    def saveConfig(self, cfg: dict) -> None:
        """
        Saves configuration dictionary `cfg` in a config.json file.

        Args:
          `cfg`: Configuration dictionary

        """

        def _serialize(obj: object):
            """Serializes data to be able to utilize json format.
            """
            if isinstance(obj, (int, float, str)):
                return obj

            elif isinstance(obj, (list, tuple)):
                return [_serialize(o) for o in obj]

            elif isinstance(obj, dict):
                newobj = {}
                for k in obj:
                    newobj[k] = _serialize(obj[k])
                return newobj

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, types.FunctionType):
                # Loss functions in lib.losses.py
                return obj.__name__

            elif isinstance(obj, (type)):
                if obj.__module__.startswith("torch.optim."):
                    # Optimizers
                    return obj.__name__

                # This includes data in lib.data
                newobj = {}
                attributes = inspect.getmembers(obj, lambda x:not(inspect.isroutine(x)))
                attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
                for name, att in attributes:
                    newobj[name] = _serialize(att)
                return newobj

            elif isinstance(obj, type(None)):
                return "None"

            elif hasattr(obj, "__module__"):
                if obj.__module__.startswith("lib.models."):
                    # Models
                    newobj = {}
                    newobj["model"] = obj.__module__
                    for att in obj.params:
                        newobj[att] = _serialize(getattr(obj, att))
                    return newobj

                elif obj.__module__.startswith("torchio.transforms"):
                    newobj = {}
                    attributes = inspect.getmembers(obj, lambda x:not(inspect.isroutine(x)))
                    attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
                    for name, att in attributes:
                        newobj[name] = _serialize(att)
                    return newobj
                else:
                    print(f"Warning: The object `{obj}` of type {type(obj)} might"
                           " not have been logged properly")
                    return str(type(obj))

            else:
                print(f"Warning: The object `{obj}` of type {type(obj)} might"
                       " not have been logged properly")
                return str(type(obj))

        serialized_cfg = _serialize(cfg)
        del serialized_cfg["data"]["opt"] # To avoid logging duplicate info

        with open(self.path, "w") as f:
            f.write(json.dumps(serialized_cfg))

        print("\n### SAVED CONFIGURATION ###\n")
        print(serialized_cfg)

def he_normal(w: TorchParameter):
    """
    He normal initialization.

    Args:
      `w` (torch.Tensor): Weights.

    Returns:
      Normal distribution following He initialization.
    """

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    return torch.nn.init.normal_(w, 0, np.sqrt(2/fan_in))


def scaleHalfGroundTruth(y_true: Tensor) -> Tensor:
    """
    Used for Deep Supervision. It halfs the size (H,W,D) of the ground truth.

    Args:
      `y_true`: Tensor containing the ground truth.

    Returns:
      Tensor with half resolution as `y_true`.
    """
    dd = [torch.linspace(-1, 1, i//2) for i in y_true.shape[2:]]
    mesh = torch.meshgrid(dd)
    grid = torch.stack(mesh, -1).cuda()
    grid = torch.stack([grid for _ in range(y_true.shape[0])])
    try:
        resized = torch.nn.functional.grid_sample(y_true, grid, mode="nearest")
    except:
        from IPython import embed; embed()
        raise Exception("para")
        pass
    return resized

def softmax2onehot(image: np.array) -> np.array:
    """
    Convert a softmax probability matrix into a onehot-encoded matrix.

    Args:
      `image` (np.array): CHWD

    Returns:
      One-hot encoded matrix.
    """
    result = np.zeros_like(image)
    labels = np.argmax(image, axis=0)
    for i in range(image.shape[0]):
        result[i] = labels==i
    return result

def sigmoid2onehot(image: np.array) -> np.array:
    """
    Convert a sigmoid probability matrix into a onehot-encoded matrix.
    The difference with softmax prob. matrices is that sigmoid allows
    labels to overlap, i.e., pixels can have multiple labels.

    Args:
      `image` (np.array): CHWD

    Returns:
      One-hot encoded matrix.
    """
    thr = 0.5
    result = 1.0*(image > thr)
    return result

def resample(image_path: str="", label_path: str="",
        voxres: Tuple[float]=(), size: List[int]=[]) -> Union[List[nib.Nifti1Image],
                                                      nib.Nifti1Image]:
    """
    Resamples an image (and its label if provided) into a specific voxel
    resolution or image size. This function is used for pre- and postprocessing
    and it can be used in two different ways:
     - Option 1: Give image_path, label_path and voxres (resample).
     - Option 2: Give label_path and size (resize).
    The raison d'être of this function is to provide with a single interface
    for resampling and resizing, which became necessary as 'resampling back to
    the original space' did not yield the same image size as the original
    images. Thus, for preprocessing, resampling is used, and, for
    postprocessing, resizing is used.

    Why I'm passing the path instead of the image? TorchIO.

    Args:
      `image_path`: Location of the image to be resampled/resized.
      `label_path`: Location of the ground truth or prediction.
      `voxres`: Voxel resolution. If len != 3, torchio might complain.
      `size`: Image dimensions. If len != 3, torchio might complain.

    Returns:
      Either a list with the image and its ground truth resampled, or
      the prediction resized.

    """
    raise Exception("Deprecated. Use 'resamplev2'")

    if label_path != "" and not os.path.isfile(label_path):
        raise ValueError(f"label_path `{label_path}` does not exist.")
    if image_path != "" and not os.path.isfile(image_path):
        raise ValueError(f"image_path `{image_path}` does not exist.")

    params = {}
    if image_path != "":
        params["im"] = tio.ScalarImage(image_path)
        print(params["im"])
    if label_path != "":
        params["label"] = tio.LabelMap(label_path)

    subject = [tio.Subject(**params)]
    if len(voxres) > 0:
        trans = tio.Resample(voxres)
    else:
        trans = tio.transforms.Resize(size, image_interpolation="nearest")
        if image_path != "":
            raise ValueError("Resize (with 'size') can only be used for labels")

    transforms = tio.Compose([trans])
    sd = tio.SubjectsDataset(subject, transform=transforms)
    loader = DataLoader(sd, batch_size=1, num_workers=4)

    results = []
    sub = list(loader)[0]
    if image_path != "":
        im = sub["im"]["data"][0,0].detach().cpu().numpy()
        aff = sub["im"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(im, affine=aff) )

    if label_path != "":
        seg = sub["label"]["data"][0,0].detach().cpu().numpy()
        seg_aff = sub["label"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(seg, affine=seg_aff) )

    if len(results) == 1:
        return results[0]
    return results

def resamplev2(images: Union[List[tio.Image], tio.Image],
        voxres: Tuple[float]=(),
        size: List[int]=[]) -> Union[List[nib.Nifti1Image], nib.Nifti1Image]:
    """
    Resamples an image (and its label if provided) into a specific voxel
    resolution or image size. This function is used for pre- and postprocessing
    and it can be used in two different ways:
     - Option 1: Give image_path, label_path and voxres (resample).
     - Option 2: Give label_path and size (resize).
    The raison d'être of this function is to provide with a single interface
    for resampling and resizing, which became necessary as 'resampling back to
    the original space' did not yield the same image size as the original
    images. Thus, for preprocessing, resampling is used, and, for
    postprocessing, resizing is used.

    Why I'm passing the path instead of the image? TorchIO.

    Args:
      `image_path`: Location of the image to be resampled/resized.
      `label_path`: Location of the ground truth or prediction.
      `voxres`: Voxel resolution. If len != 3, torchio might complain.
      `size`: Image dimensions. If len != 3, torchio might complain.

    Returns:
      Either a list with the image and its ground truth resampled, or
      the prediction resized.

    """

    if len(voxres) == len(size) == 0:
        raise Exception("Either 'voxres' or 'size' must be indicated")
    if len(voxres) != 0 and len(size) != 0:
        raise Exception("Either 'voxres' or 'size' must be indicated (not both)")
    if len(voxres) == 0 and len(size) != 3:
        raise Exception("'size' should have only 3 elements")
    if len(size) == 0 and len(voxres) != 3:
        raise Exception("'voxres' should have only 3 elements")

    if not isinstance(images, list):
        images = [images]

    for im in images:
        if not isinstance(im, tio.Image):
            raise Exception("'images' expected to be tio.Images...")

    params = {}
    scalars, labels = 0, 0
    for im in images:
        if isinstance(im, tio.ScalarImage):
            scalars += 1
            params[f"im_{scalars}"] = im
        elif isinstance(im, tio.LabelMap):
            labels += 1
            params[f"label_{labels}"] = im

    subject = [tio.Subject(**params)]
    if len(voxres) > 0:
        # This typically happens in the preprocessing, when we have images and labels
        trans = tio.Resample(voxres)
    else:
        # Here, we typically have a single image for postprocessing.
        if len(images) > 1:
            print("WARNING: We are going to use interpolation=linear. "
                  "If there are LabelMaps, I think that they are interpolated "
                  "automatically with 'nearest'. Check!")
            #trans = tio.transforms.Resize(size, image_interpolation="linear")
        trans = tio.transforms.Resize(size, image_interpolation="linear")

    transforms = tio.Compose([trans])
    sd = tio.SubjectsDataset(subject, transform=transforms)
    loader = DataLoader(sd, batch_size=1, num_workers=4)

    results = []
    sub = list(loader)[0]
    for i in range(1, scalars+1):
        im = sub[f"im_{i}"]["data"][0].detach().cpu().numpy()
        aff = sub[f"im_{i}"]["affine"][0].detach().cpu().numpy()
        if im.shape[0] == 1:
            im = im[0]
        results.append( nib.Nifti1Image(im, affine=aff) )

    for i in range(1, labels+1):
        seg = sub[f"label_{i}"]["data"][0,0].detach().cpu().numpy()
        seg_aff = sub[f"label_{i}"]["affine"][0].detach().cpu().numpy()
        results.append( nib.Nifti1Image(seg, affine=seg_aff) )

    if len(results) == 1:
        return results[0]
    return results

# FROM CENTRIPETAL SGD PAPER
def generate_merge_matrix_for_kernel(deps, layer_idx_to_clusters, kernel_namedvalue_list):
    result = {}
    #from IPython import embed; embed()
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        merge_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            if len(clst) == 1:
                merge_trans_mat[clst[0], clst[0]] = 1
                continue
            sc = sorted(clst)       # Ideally, clst should have already been sorted in ascending order
            for ei in sc:
                for ej in sc:
                    merge_trans_mat[ei, ej] = 1 / len(clst)
        result[kernel_namedvalue_list[layer_idx].name] = torch.from_numpy(merge_trans_mat).cuda()
    return result

def generate_decay_matrix_for_kernel_and_vecs(deps, layer_idx_to_clusters, kernel_namedvalue_list, weight_decay, weight_decay_bias, centri_strength):
    KERNEL_KEYWORD = 'conv.weight'
    result = {}
    #   for the kernel
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                decay_trans_mat[ee, ee] = weight_decay + centri_strength
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength / len(clst)
        kernel_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name] = kernel_mat

    #   for the vec params (bias, beta and gamma), we use 0.1 * centripetal strength
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                # Note: using smaller centripetal strength on the scaling factor of BN improve the performance in some of the cases
                decay_trans_mat[ee, ee] = weight_decay_bias + centri_strength * 0.1
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength * 0.1 / len(clst)
        vec_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.weight')] = vec_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.bias')] = vec_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'conv.bias')] = vec_mat

    return result

def get_layer_idx_to_clusters(kernel_namedvalue_dict, target_deps):
    # Implement somehow the same thing as with pacesetter_dict
    # pacesetter_dict seems to be for avoiding non-convolutional layers
    result = {}
    for named_kv in kernel_namedvalue_dict:
        num_filters = kernel_namedvalue_dict[named_kv].value.shape[0]
        layer_idx = named_kv
        #from IPython import embed; embed()
        #if pacesetter_dict is not None and _is_follower(layer_idx, pacesetter_dict):
        #    continue

        if num_filters > target_deps[layer_idx]:
            result[layer_idx] = cluster_by_kmeans(kernel_value=kernel_namedvalue_dict[named_kv].value, num_cluster=target_deps[layer_idx], layer_idx=layer_idx)
        elif num_filters < target_deps[layer_idx]:
            raise ValueError('wrong target dep')

    return result

def get_all_conv_kernel_namedvalue_as_dict(self, dims):
    from collections import namedtuple
    from lib.models.Sauron import DropChannels

    NamedParamValue = namedtuple('NamedParamValue', ['name', 'value'])

    result = {}
    #for k, v in self.state_dict().items():
    #    if v.dim() == dims:
    #        result.append(NamedParamValue(name=k, value=v.cpu().numpy()))
    for mod in self.modules():
        if isinstance(mod, DropChannels):
            for inner_mod in mod.module.modules():
                if isinstance(inner_mod, (torch.nn.Conv2d, torch.nn.Conv3d,
                    torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                    result[mod.name] = NamedParamValue(name=mod.name,
                                    value=inner_mod.weight.cpu().detach().numpy())
    return result

def cluster_by_kmeans(kernel_value, num_cluster, layer_idx):
    # In Pytorch, the output filters in transposed convs are located in axis=1
    if "Transpose" in layer_idx:
        ax = 1
    else:
        ax = 0
    #assert kernel_value.ndim == 4 # Remove this because 3D convs are of ndim=5
    x = np.reshape(kernel_value, (kernel_value.shape[ax], -1))
    if num_cluster == x.shape[0]:
        result = [[i] for i in range(num_cluster)]
        return result
    else:
        print('cluster {} filters into {} clusters'.format(x.shape[0], num_cluster))
    km = KMeans(n_clusters=num_cluster)
    km.fit(x)
    result = []
    for j in range(num_cluster):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result

def get_all_deps(kernel_namedvalue_dict):
    # My own function. To retrieve the number of filters of the layers
    all_filters = {}
    for named_kv in kernel_namedvalue_dict:
        # In Pytorch, the output filters in TransposedConvs are in axis=1
        if "Transpose" in named_kv:
            num_filters = kernel_namedvalue_dict[named_kv].value.shape[1]
        else:
            num_filters = kernel_namedvalue_dict[named_kv].value.shape[0]
        all_filters[named_kv] = num_filters
    return all_filters
