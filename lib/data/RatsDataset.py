import torch, os, random, time
import nibabel as nib
import numpy as np
from lib.data.BaseDataset import BaseDataset
import torchio as tio
import lib.loss as loss
from typing import List
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import InstanceNorm2d
from lib.models.UNet import UNet
from lib.models.nnUNet import nnUNet
from lib.models.Sauron import Sauron


class Double2FloatTransform(tio.transforms.Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        im = subject["t2"]
        lab = subject["label"]
        X = im.data
        Y = lab.data

        X = X.type(torch.float32)
        Y = Y.type(torch.float32)

        im.set_data(X)
        lab.set_data(Y)
        return subject

class RatsDataset(BaseDataset):
    """
    Lesion segmentation with one image. Toy dataset.
    """
    # Dataset info
    name = "rats"
    modalities = 1
    classes = {0: "background", 1: "lesion"}
    dim = "2D"
    # Specify how the final prob. values are computed: softmax or sigmoid?
    onehot = "softmax"
    # Which classes will be reported during validation
    measure_classes_mean = np.array([1])
    #data_path = "/home/miguelv/data/in/CR_DATAv2/"
    #data_path = "/media/miguelv/HD1/Datasets/CR_DATAv2/" #FUJ
    data_path = ""
    infoX = ["t2"] # Image modalities
    infoY = ["label"] # Labels
    infoW = [] # Optional Weights

    # Sauron properties. Leave here for logging.
    dist_fun = "euc_norm"
    imp_fun = "euc_rand"
    sf = 2

    # Optimization strategy
    opt = {"architecture":
                {"modalities": modalities, "n_classes": len(classes),
                    "dist_fun": dist_fun, "imp_fun": imp_fun, "sf": sf,
                    "fms_init": 32, "levels": 5, "normLayer": InstanceNorm2d, # orginal: 32
                    "dim": "2D"},
            "loss": loss.DS_CrossEntropyDiceLoss_Distance,
            "batch_size": 4,
            "epochs": 1000,
            #"optim": torch.optim.Adam,
            "optim": torch.optim.SGD,
            "optim_params": {"lr": 1e-2, "weight_decay": 1e-5, "momentum": 0.9},
            "scheduler": LambdaLR, # Polynomial learning rate decay
            "scheduler_params": {"lr_lambda": lambda ep: (1 - ep/1000)**0.9}
            }

    # Data augmentation strategy

    # All transformations look realistic, and even a bit conservative
    # Random scale: resize image between 0.9 and 1.1 in the X-Y plane
    # Random rotation: rotate between -10 and 10 degrees in the X-Y plane
    # Random translation: move the image in the X-Y plane; max: 10%.
    transforms_train = tio.Compose([
        Double2FloatTransform(),
        tio.transforms.RandomAffine(scales=[0.9, 1.1, 0.9, 1.1, 1, 1],
            degrees=[0, 0, 0, 0, 0, 0], translation=[0, 0, 0, 0, 0, 0], p=0.5),
        tio.transforms.RandomAffine(scales=[1, 1, 1, 1, 1, 1],
            degrees=[0, 0, 0, 0, -10, 10], translation=[0, 0, 0, 0, 0, 0], p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
        tio.RandomFlip(axes=(0,), flip_probability=0.5),
        tio.RandomFlip(axes=(1,), flip_probability=0.5),
        tio.transforms.OneHot(num_classes=len(classes)),
        tio.transforms.ZNormalization(),
        ])
    transforms_val = tio.Compose([
        Double2FloatTransform(),
        tio.transforms.OneHot(num_classes=len(classes)),
        tio.transforms.ZNormalization(),
        ])
    transforms_test = tio.Compose([
        Double2FloatTransform(),
        tio.transforms.OneHot(num_classes=len(classes)),
        tio.transforms.ZNormalization(),
        ])

    def __init__(self, split: List[float], seed: int):
        """
        Divide the data into train/validation splits.

        Args:
          `split`: Percentage of training data.
        """
        self.transforms_dict = {"train": self.transforms_train,
                "validation": self.transforms_val, "test": self.transforms_test}

        studies = ["21JUL2015"]
        print("Loading data...")

        # Collecting the files in a list to read them when need it
        brains_all = []
        for study in studies:
            for root, subdirs, files in os.walk(self.data_path + study + "/"):
                if "scan_lesion.nii.gz" in files:
                    timepoint = study + "_" + root.split("/")[-2]
                    # Remove 2h, 48h -> very few data -> noisy results
                    if timepoint in ["2h", "48h"]:
                      continue

                    brains_all.append(root + "/")

        brains_all = sorted(brains_all)
        # Test is 20% of the data
        brains_test = brains_all[:int(0.2*len(brains_all))]
        brains_all = brains_all[int(0.2*len(brains_all)):]

        random.seed(seed)
        random.shuffle(brains_all)

        s1 = int(len(brains_all)*split[0])
        brains_train = brains_all[:s1]
        brains_val = brains_all[s1:]

        # Temporal thing
        #brains_train = brains_all[0:1]
        #brains_val = brains_train
        #brains_test = brains_train

        self.subjects_dict = {"train": [], "validation": [], "test": []}

        for brain_list, partition in zip([brains_train, brains_val, brains_test], ["train", "validation", "test"]):
            for brain_path in brain_list:
                study, timepoint, subject = brain_path.split("/")[-4:-1]
                id_ = study + "_" + timepoint + "_" + subject
                brain_list_size = nib.load(brain_path + "scan_lesion.nii.gz").get_fdata().shape
                slices = brain_list_size[-1]

                self.subjects_dict[partition].append(tio.Subject(
                    t2=tio.ScalarImage(brain_path + "scan.nii.gz"),
                    label=tio.LabelMap(brain_path + "scan_lesion.nii.gz"),
                    info={
                        "voxelspacing": [30/256, 30/256, 1],
                        "id": id_,
                        "path": brain_path + "scan.nii.gz",
                        "slices": slices,
                        "patch_size": (256, 256, 1)
                        }
                    ))

        print("Training images", len(self.subjects_dict["train"]))
        print("Validation images", len(self.subjects_dict["validation"]))
        print("Test images", len(self.subjects_dict["test"]))

    def save(self, pred: np.array, affine: np.array, header: nib.Nifti1Header,
            path_output: str) -> None:
        """
        Save `pred` in `path_output`.

        Args:
          `pred`: Shape CWHD.
          `affine`: 4x4 matrix.
          `header`: Nifti header.
          `path_output`: Location where the image will be saved.
        """
        pred = np.argmax(pred, axis=0)
        res_im = nib.Nifti1Image(pred, affine=affine, header=header)
        #nib.save(image, path_output)

        # Without the following line, predictions become quite heavy
        # and the online platform will throw an error:
        # "The container was killed as it exceeded the memory limit of 4g."
        res_im.header.set_data_dtype(np.dtype("uint8"))
        #output_path =  os.path.join(outputFolder,
        #        f"{folder}.nii.gz".replace("case", "prediction"))
        nib.save(res_im, path_output)

        # For some reason, this won't save the labels correctly
        # Instead of [0, 1, 2] -> [0, 0.9999, 2.00001]
        # So, we do the following:
        tt = nib.load(path_output)
        newim = nib.Nifti1Image(nib.casting.float_to_int(tt.get_fdata(),
                                                int_type=np.dtype("uint8")),
                                affine=tt.affine, header=tt.header)
        nib.save(newim, path_output)

    '''
    def save(self, pred: np.array, path_output: str, original_im_path: str):
        """
        Save `pred` in `path_output`.

        Args:
          `pred`: Shape CWHD.
          `path_output`: Directory where predictions will be saved.
          `original_im_path`: Path to find the original image.
        """
        pred = np.argmax(pred, axis=0)
        orig_im = nib.load(original_im_path)
        nib.save(nib.Nifti1Image(pred, affine=orig_im.affine,
            header=orig_im.header), path_output)
    '''
    @staticmethod
    def findGroundTruthFiles(path, _):
        #return [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith(".nii.gz") ]
        brains_all = []
        study = "21JUL2015"
        for root, subdirs, files in os.walk(os.path.join(path, study, "24h")):
            if "scan_lesion.nii.gz" in files:
                #brains_all.append(root + "/")
                #brains_all.append( "_".join(root.split("/")[-3:])+".nii.gz" )
                brains_all.append( os.path.join(root, "scan_lesion.nii.gz") )

        brains_all = sorted(brains_all)
        # Test is 20% of the data
        brains_test = brains_all[:int(0.2*len(brains_all))]
        return brains_test

    @staticmethod
    def findPredictionFiles(path):
        preds = [os.path.join(path, x) for x in sorted(os.listdir(path))]
        return preds

    @staticmethod
    def pre_verify(inputFolder: str) -> bool:
        return True
    @staticmethod
    def pre_process(inputFolder: str, outputFolder: str, _=None) -> None:
        return True
    @staticmethod
    def post_process(image: np.array, subject: tio.data.subject.Subject,
            path_original: str) -> nib.Nifti1Image:

        """
        case = "/".join(subject["info"]["path"].split("/")[-2:])
        path_original = os.path.join(path_original, case)
        im_orig = nib.load(path_original)
        im_preprocessed = nib.load(subject["info"]["path"])

        images = [tio.ScalarImage(
                    tensor=image,
                    affine=im_preprocessed.affine)]
        image = utils.resamplev2(images=images,
                      size=im_orig.shape)

        image = nib.Nifti1Image(image.get_fdata(), affine=im_orig.affine,
                                header=im_orig.header)

        return image
        """
        raise Exception("not in use")

    @staticmethod
    def finalArrangements(predFolder: str, origFolder: str) -> None:
        pass

