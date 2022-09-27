import numpy as np
from skimage import measure
from scipy import ndimage
from medpy import metric
import torch
from typing import List, Callable, Dict
from lib.metric_utils import compute_surface_distances, compute_surface_dice_at_tolerance
from lib.utils import softmax2onehot

def surface_dice_np(pred, true, voxres):
    """
    Compute the surface dice coefficient.
    Source: https://www.jmir.org/2021/7/e26151
    Intuition: two segmentations with very similar surface are probably equally
    good. A different surface would require in practice manual intervention.
    How similar these surfaces (pred, true) can be depends on the tolerance.
    A very low tolerance penalizes more slightly different surfaces.

    Args:
      `pred`: Prediction (HW(D)).
      `true`: Label (HW(D)).
      `voxres`: Voxel resolution.

    Returns:
      Surface dice.

    About the tolerance:
    "We computed these acceptable tolerances for each organ by measuring the
    interobserver variation in segmentations between 3 different consultant
    oncologists (each with over 10 years of experience in organ at risk
    delineation) on the validation subset of TCIA images."
    (So... the paper doesn't really specify *how* it was computed)
    """
    tolerance = 1.64 # Default tolerance
    surface_distances = compute_surface_distances(true.astype(bool),
              pred.astype(bool), voxres)
    return compute_surface_dice_at_tolerance(surface_distances, tolerance)



def border_np(y: np.array) -> np.array:
    """
    Compute the border of a binary mask.

    Args:
      `y`: Binary mask.

    Returns:
      Border of `y`.
    """
    return y - ndimage.binary_erosion(y)

def iou_np(pred: np.array, true: np.array) -> float:
    """
    Compute the Intersection over Union.

    Args:
      `pred`: Prediction (HW(D)).
      `true`: Label (HW(D)).

    Returns:
      Intersection over union.
    """
    intersection = np.sum(pred * true)
    union = np.sum((pred + true) != 0)
    return intersection / union

def compactness_np(pred: np.array) -> float:
    """
    Compute the compactness.

    Args:
      `pred`: Prediction (HW(D)).
      `true`: Label (HW(D)).

    Returns:
      Compactness.
    """
    area = np.sum(border_np(pred))
    volume = np.sum(pred)
    return area**1.5/volume

class Metric:
    def __init__(self, metrics: List[str], onehot: Callable,
            classes: Dict[int, str], classes_mean: List[int],
            multiprocess: bool):
        """

        Args:
          `metrics`: Metrics that will be measured, e.g., dice.
          `onehot`: Function to convert into a onehot encoded matrix.
              This is of special importance since softmax/sigmoid activations
              in the last layer produce mutually and non-mutually exclusive
              labels, and this needs to be considered when producing the masks
              and, consequently, when assessing them.
          `classes`: Classes that will be measured.
          `classes_mean`: Classes over which the mean will be computed (for
              validation purposes)
          `multiprocess`: Whether to use multiprocess.
        """
        self.metrics = metrics
        if onehot == "softmax":
            self.onehot = softmax2onehot
        else:
            raise ValueError(f"Unknown value of 'onehot': {onehot}")
        self.classes = classes
        self.classes_mean = classes_mean
        self.multiprocess = multiprocess


        if self.multiprocess:
            import multiprocessing
            self.pool = multiprocessing.Pool(processes=8)

    def all(self, y_pred: np.array, y_true: np.array, info: dict):
        """
        Helper function. Computes the metrics either with or without
        multiprocessing, and it saves the voxelspacing (for HD).

        Args:
          `y_pred`: Prediction (BWH(D))
          `y_pred`: Label (BWH(D))

        Returns:
            Either the computed results or a "pooled instance".
        """
        self.voxelspacing = info["voxelspacing"]

        if self.multiprocess:
            return self.pool.apply_async(self.all_, args=(y_pred, y_true))
        else:
            return self.all_(y_pred, y_true)

    def all_(self, y_pred: np.array, y_true: np.array) -> dict:
        """
        Computes all metrics.

        Args:
          `y_pred`: Prediction (BWH(D))
          `y_pred`: Label (BWH(D))

        Returns:
            Dictionary with keys=metrics, values=results.
        """
        results = {}
        for metric in dir(self):
            if metric in self.metrics:
                # Given the metric name, execute it and get the result
                # Return result from each function will be a dictionary
                # of the form: {"metric_name": [result]}
                results.update(getattr(self, metric)(y_pred, y_true))

        return results

    def _getMean(self, results: np.array) -> List[float]:
        """
        Gets the average of the results. If matrix `results` contains -1
        it means that such class in such sample was not found in the ground
        truth, and therefore, the current metric was not computed.

        This function calculates the mean of the computed results accounting
        for labels and cases that were not computed, which will be -1.

        Args:
          `results` (np.array): Matrix of size n_samples, n_classes

        Returns:
          List of size `n_classes` with the mean metric per class
          accounting for those cases where such class was found. If
          no samples had such class, the it produces a value of -1.
        """
        final_results = []
        for c in sorted(self.classes):
            elem = results[c][results[c] != -1]
            if len(elem) == 0:
                final_results.append(-1)
            else:
                final_results.append(elem.mean())

        return final_results

    def dice(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        This function calculates the Dice coefficient.
        Works for 2D and 3D images.
        Input size: CHWD

        Returns:
           List with Dice coefficients. len = C
        """
        # -1 to know in which classes this metric was not computed
        results = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            true = 1.0*(y_true_all[c] > 0.5)
            if np.sum(true) > 0: # If class c is in the ground truth
                results[c] = metric.dc(pred, true)

        return {"dice": self._getMean(results)}

    def iou(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        This function calculates the Intersection over Union (Jaccard Index).
        Works for 2D and 3D images.
        Input size: CHWD

        Returns:
           List with Dice coefficients. len = C
        """
        results = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            true = 1.0*(y_true_all[c] > 0.5)
            if np.sum(true) > 0: # If class c is in the ground truth
                results[c] = iou_np(pred, true)

        return {"iou": self._getMean(results)}

    def TFPN(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        True and False Positives and Negatives.
        Works for 2D and 3D images.
        Input size: CHWD

        Returns:
           List with Dice coefficients. len = C
        """
        results1 = np.zeros((len(self.classes))) - 1
        results2 = np.zeros((len(self.classes))) - 1
        results3 = np.zeros((len(self.classes))) - 1
        results4 = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            true = 1.0*(y_true_all[c] > 0.5)
            results1[c] = np.sum(pred * true)
            results2[c] = np.sum((1-pred) * (1-true))
            results3[c] = np.sum(pred * (1-true))
            results4[c] = np.sum((1-pred) * true)

        return {"TP": self._getMean(results1),
                "TN": self._getMean(results2),
                "FP": self._getMean(results3),
                "FN": self._getMean(results4)}

    def HD(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        Hausdorff distance.
        """
        results1 = np.zeros((len(self.classes))) - 1
        results2 = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            if pred.sum() == 0:
              pred += 1.0
            true = 1.0*(y_true_all[c] > 0.5)
            if np.sum(true) > 0:
                results1[c] = metric.hd(pred, true, self.voxelspacing)
                results2[c] = metric.hd95(pred, true, self.voxelspacing)

        return {"HD": self._getMean(results1),
                "HD95": self._getMean(results2)}

    def compactness(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        surface^1.5 / volume
        """
        results = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            if np.sum(pred) > 0:
                results[c] = compactness_np(pred)

        return {"compactness": self._getMean(results)}

    def surface_dice(self, y_pred_all: np.array,
            y_true_all: np.array) -> List[Dict[str, List]]:
        """
        This function calculates the Dice coefficient.
        Works for 2D and 3D images.
        Input size: CHWD

        Returns:
           List with Dice coefficients. len = C
        """
        # -1 to know in which classes this metric was not computed
        results = np.zeros((len(self.classes))) - 1
        y_pred = self.onehot(y_pred_all)
        for c in sorted(self.classes):
            pred = 1.0*(y_pred[c] > 0.5)
            true = 1.0*(y_true_all[c] > 0.5)
            if np.sum(true) > 0: # If class c is in the ground truth
                results[c] = surface_dice_np(pred, true, self.voxelspacing)

        return {"surface_dice": self._getMean(results)}

    def __getstate__(self):
        """
        Executed when doing.get(). If we don't delete the pool
        it will return an Exception.
        Read more here: https://stackoverflow.com/questions/25382455/python-notimplementederror-pool-objects-cannot-be-passed-between-processes/25385582
        """
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def close(self) -> None:
        """
        Closes the pool used for multiprocessing.
        """
        if self.multiprocess:
            self.pool.close()
            self.pool.join()
            self.pool.terminate()

    def getMeanValScores(self, results: dict) -> str:
        """
        Compute the average of each metric in the classes initially given.
        Used during validation.

        Args:
          `results`: Contains all results.

        Returns:
          String with the corresponding metric and its average value.
        """
        t = ""
        for m in self.metrics:
            val = [np.array(results[k][m])[self.classes_mean] for k in results]
            val = np.round(np.mean(val), 5)
            t += f"Val {m}: {val}. "

        return t[:-2]

    def getAllValScores(self, results: dict) -> str:
        """
        Retrieves in form of text the validation scores.
        """
        t = ""
        metrics = results[list(results.keys())[0]].keys()
        for m in metrics:
            val = np.array([np.array(results[k][m])[self.classes_mean] for k in results])
            val = [np.round(x, 5) for x in val.mean(axis=0)]
            t += f"Val {m}: {val}. "

        return t[:-2]

