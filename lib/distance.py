import torch
from torch import Tensor

def Euclidean_norm(fm: Tensor, compress) -> Tensor:
    """
    Computes the Euclidean distance w.r.t. the first channel, and normalizes
    the distances.
    """
    fm1 = fm[:, 0:1]
    fm2 = fm[:, 1:]

    fm1_max_vals = torch.amax(fm1, axis=[2,3], keepdim=True)
    fm1_min_vals = torch.amin(fm1, axis=[2,3], keepdim=True)
    fm1_norm = (fm1-fm1_min_vals) / (fm1_max_vals - fm1_min_vals)

    fm2_max_vals = torch.amax(fm2, axis=[2,3], keepdim=True)
    fm2_min_vals = torch.amin(fm2, axis=[2,3], keepdim=True)
    fm2_norm = (fm2-fm2_min_vals) / (fm2_max_vals - fm2_min_vals)

    diff = compress(fm1_norm) - compress(fm2_norm)

    L2_norm = torch.sqrt( torch.mean( torch.pow(diff, 2) , axis=(2, 3)) )
    return L2_norm

def Euclidean_rand(fm: Tensor, compress) -> Tensor:
    """
    Computes the Euclidean distance w.r.t. a random feature map channel.
    """
    ref_idx = torch.randint(low=0, high=fm.shape[1], size=(1,1))[0]

    fm1 = fm[:, ref_idx:ref_idx+1]
    fm2 = fm
    diff = compress(fm1) - compress(fm2)
    axis = [i for i in range(2, len(fm.shape))] # 2D: [2,3], 3D: [2,3,4]
    L2_norm = torch.sqrt( torch.mean( torch.pow(diff, 2) , axis=axis) )

    max_vals = torch.max(L2_norm, axis=1, keepdim=True)[0] # Size (B, C)
    distance = L2_norm / (max_vals + 1e-15)

    avg_distance_across_batches = torch.mean(distance, axis=0)

    return avg_distance_across_batches, ref_idx


def Euclidean_norm_deltaprunenorm(fm: Tensor, compress) -> Tensor:
    """
    Computes the Euclidean distance w.r.t. the first channel, and normalizes
    the distances.
    Normalizes the feature maps as in Euclidean rand.
    """
    fm1 = fm[:, 0:1]
    fm2 = fm[:, 1:]

    diff = compress(fm1) - compress(fm2)
    axis = [i for i in range(2, len(fm.shape))] # 2D: [2,3], 3D: [2,3,4]
    L2_norm = torch.sqrt( torch.mean( torch.pow(diff, 2) , axis=axis) )

    max_vals = torch.max(L2_norm, axis=1, keepdim=True)[0] # Size (B, C)
    distance = L2_norm / (max_vals + 1e-15)

    return distance

def Euclidean_norm_nonorm(fm: Tensor, compress) -> Tensor:
    """
    Computes the Euclidean distance w.r.t. the first channel, and normalizes
    the distances.
    It does not normalize the feature maps in any way.
    """
    fm1 = fm[:, 0:1]
    fm2 = fm[:, 1:]

    diff = compress(fm1) - compress(fm2)
    axis = [i for i in range(2, len(fm.shape))] # 2D: [2,3], 3D: [2,3,4]
    L2_norm = torch.sqrt( torch.mean( torch.pow(diff, 2) , axis=axis) )

    return L2_norm
