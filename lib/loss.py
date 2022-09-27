# This file contains all the loss functions that I've tested, including the
# proposed "Rectified normalized Region-wise map".
#
# To simplify and for clarity reasons, I've only commented those functions
# that appear in the paper. In any case, there is a lot of repetion because
# the class "BaseData" computes the "weights" (aka Region-wise map) based
# on the name of the loss function; therefore, different loss function names
# provide different weights.
import torch
import numpy as np
from torch import Tensor
from typing import List
from lib.utils import scaleHalfGroundTruth

def CrossEntropyLoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Regular Cross Entropy loss function.
    It is possible to use weights with the shape of BWHD (no channel).

    Args:
      `y_pred`: Prediction of the model.
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    if not isinstance(y_pred, list):
        raise TypeError(f"`y_pred` should be a list")
    if not isinstance(y_true, list):
        raise TypeError(f"`y_pred` should be a list")

    y_pred, y_true = y_pred[0], y_true[0]
    if y_pred.shape != y_true.shape:
        raise ValueError(f"`y_pred` ({y_pred.shape}) and "
                         f"`y_true` ({y_true.shape}) shapes differ.")

    ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    return -torch.mean(ce)

def DiceLoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Binary Dice loss function.

    Args:
      `y_pred`: Prediction of the model.
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    if not isinstance(y_pred, list):
        raise TypeError(f"`y_pred` should be a list")
    if not isinstance(y_true, list):
        raise TypeError(f"`y_pred` should be a list")

    y_pred, y_true = y_pred[0], y_true[0]
    if y_pred.shape != y_true.shape:
        raise ValueError(f"`y_pred` ({y_pred.shape}) and "
                         f"`y_true` ({y_true.shape}) shapes differ.")

    axis = list([i for i in range(2, len(y_true.shape))]) # for 2D/3D images
    num = 2 * torch.sum(y_pred * y_true, axis=axis)
    denom = torch.sum(y_pred + y_true, axis=axis)
    return (1 - torch.mean(num / (denom + 1e-6)))

def CrossEntropyDiceLoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Combine CrossEntropy and Dice losss.

    Args:
      `y_pred`: Prediction of the model.
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    ce = CrossEntropyLoss(y_pred, y_true)
    dice = DiceLoss(y_pred, y_true)
    return ce + dice

def DS_CrossEntropyDiceLoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    CrossEntropyDice loss with deep supervision.

    Args:
      `y_pred`: Prediction of the model.
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    y_true = y_true[0]

    loss = CrossEntropyDiceLoss([y_pred[0]], [y_true])
    for pred in y_pred[1:]:
        y_true = scaleHalfGroundTruth(y_true)
        loss += CrossEntropyDiceLoss([pred], [y_true])

    return loss

def DS_CrossEntropyDiceLoss_Distance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Combine CrossEntropy, Dice losss, and minimize the distance between
    the feature maps (for pruning).

    Args:
      `y_pred`: Prediction of the model. List[List of DS, List of Distances]
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    y_true = y_true[0]
    distances = y_pred[1]
    y_pred = y_pred[0]

    loss = CrossEntropyDiceLoss([y_pred[0]], [y_true])
    for pred in y_pred[1:]:
        y_true = scaleHalfGroundTruth(y_true)
        loss += CrossEntropyDiceLoss([pred], [y_true])

    res = 0
    for distance in distances:
        res += torch.mean(distance)
    res /= len(distances)

    lambda_ = 0.5
    return loss + lambda_*res


