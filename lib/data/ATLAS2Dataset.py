import torch, os, random, time, re
import nibabel as nib
import numpy as np
from lib.data.BaseDataset import BaseDataset
from lib.loss import DS_CrossEntropyDiceLoss_Distance
from typing import List
import torchio as tio
from lib.models.Sauron import Sauron
from torch.nn import InstanceNorm3d
from torch.optim.lr_scheduler import LambdaLR
import lib.utils as utils

class ATLAS2Dataset(BaseDataset):
    """
    ATLAS V2.0 Dataset.
    Paper: https://doi.org/10.1038/s41597-022-01401-7
    """
    name = "atlas2"
    modalities = 1
    classes = {0: "nontumor", 1: "tumor"}
    dim = "3D"
    # Specify how the final prob. values are computed: softmax or sigmoid?
    onehot = "softmax"
    # Which classes will be reported during validation
    measure_classes_mean = np.array([1])
    # These values are filled by get_dataset() in lib/helpers.py
    infoX = ["im"]
    infoY = ["label"]
    infoW = []

    dist_fun = "euc_norm"
    imp_fun = "euc_rand"
    sf = 2

    # Optimization strategy
    # NOTE: fms_init=24, levels=5 fits in FUJ
    # NOTE: fms_init=32, levels=6 originally
    opt = {"architecture":
                { "modalities": modalities, "n_classes": len(classes),
                    "dist_fun": dist_fun, "imp_fun": imp_fun, "sf": sf,
                    "fms_init": 24, "levels": 5, "normLayer": InstanceNorm3d,
                    "dim": "3D"},
                "loss": DS_CrossEntropyDiceLoss_Distance,
            "batch_size": 1,
            "epochs": 500,
            "optim": torch.optim.Adam,
            "optim_params": {"lr": 1e-3, "weight_decay": 1e-5},
            "scheduler": LambdaLR, # Polynomial learning rate decay
            "scheduler_params": {"lr_lambda": lambda ep: (1 - ep/500)**0.9}
            }

    # All transformations look realistic, and even a bit conservative
    # Random scale: resize image between 0.9 and 1.1 in the X-Y plane
    # Random rotation: rotate between -10 and 10 degrees in the X-Y plane
    # Random translation: move the image in the X-Y plane; max: 10%.
    transforms_train = tio.Compose([
            tio.transforms.RandomAffine(scales=[0.85, 1.25, 0.85, 1.25, 1, 1], degrees=[0, 0, 0, 0, 0, 0], translation=[0, 0, 0, 0, 0, 0], p=0.2),
            tio.transforms.RandomAffine(scales=[1, 1, 1, 1, 1, 1], degrees=[0, 0, 0, 0, -180, 180], translation=[0, 0, 0, 0, 0, 0], p=0.2), # or even 30 degrees

            tio.RandomGamma(log_gamma=(-0.3, 0.5), p=0.3), # (0.7,1.5)
            tio.RandomFlip(axes=(0,), flip_probability=0.5),
            tio.RandomFlip(axes=(1,), flip_probability=0.5),
            tio.RandomFlip(axes=(2,), flip_probability=0.5),
            # control points = 14 is not bad neither
            tio.transforms.RandomElasticDeformation(num_control_points=7,
                locked_borders=2, p=0.3),
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])
    transforms_val = tio.Compose([
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])
    transforms_test = tio.Compose([
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])

    #sampler = tio.data.LabelSampler(
    #        patch_size=(80, 160, 160),
    #        label_name="label",
    #        label_probabilities={0:0.01, 1:1, 2:2})

    def __init__(self, split: List[float], seed: int):
        """Prepares the data into self.list

           Args:
            `split`: Percentage of data used for training and validation.
        """
        self.transforms_dict = {"train": self.transforms_train,
                "validation": self.transforms_val,
                "test": self.transforms_test}


        print("Loading data...")
        print("NOTE: This differs from nnUNet because it has one level less")
        t0 = time.time()

        ids = []
        for root, subdirs, files in os.walk(self.data_path):
            for f in files:
                if f.endswith(".nii.gz"):
                    ids.append( "_".join(root.split("/")[-4:-2]) )
                    break # Because there are 2 files, the imgs and the labels
        ids = sorted(ids)
        # Test is 20% of the data
        test_ids = ids[:int(0.2*(len(ids)))]
        tr_ids = ids[int(0.2*(len(ids))):]

        random.seed(seed)
        random.shuffle(ids)

        s1 = int(len(tr_ids)*split[0])
        val_ids = tr_ids[s1:]
        tr_ids = tr_ids[:s1]

        # Test overfitting
        #tr_ids = ids[0:1]
        #val_ids = ids[0:1]
        #test_ids = ids[0:1]

        self.subjects_dict = {"train": [], "validation": [], "test": []}
        for ids, partition in zip([tr_ids, val_ids, test_ids], ["train", "validation", "test"]):
            for im_id in ids:
                cohort, subject = im_id.split("_")
                pp = os.path.join(self.data_path, cohort, subject, "ses-1", "anat")

                image_path = os.path.join(
                        pp, f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz")
                label_path = os.path.join(
                        pp, f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")

                tmp_im = nib.load(image_path)
                spacing = np.array(tmp_im.header.get_zooms())
                slices = 2

                self.subjects_dict[partition].append(tio.Subject(
                    im=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                    info={
                        "voxelspacing": spacing,
                        "id": im_id,
                        "path": image_path,
                        "slices": slices,
                        "patch_size": (128, 128, 128)
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

    ### FOR PREPROCESSING AND POSTPROCESSING
    @staticmethod
    def pre_verify(inputFolder: str) -> bool:
        """
        """
        raise Exception("Not in use")


    @staticmethod
    def pre_process(inputFolder: str, outputFolder: str, _=None) -> None:
        """
        """
        raise Exception("Not in use")

    @staticmethod
    def post_process(image: np.array, subject: tio.data.subject.Subject,
            path_original: str) -> nib.Nifti1Image:
        """
        """
        raise Exception("Not in use")

    @staticmethod
    def post_process_old(inputFolder: str, outputFolder: str,
                     original: str) -> None:
        """
        """
        raise Exception("deprecated")

    @staticmethod
    def findGroundTruthFiles(path, _):
        #raise NotImplementedError("para")
        ids = []
        for root, subdirs, files in os.walk(path):
            for f in files:
                if f.endswith(".nii.gz"):
                    ids.append( "_".join(root.split("/")[-4:-2]) )
                    break # Because there are 2 files, the imgs and the labels
        ids = sorted(ids)
        # Test is 20% of the data
        test_ids = ids[:int(0.2*(len(ids)))]
        tr_ids = ids[int(0.2*(len(ids))):]

        gt_files = []
        for tid in test_ids:
            cohort, subject = tid.split("_")
            pp = os.path.join(path, cohort, subject, "ses-1", "anat")

            label_path = os.path.join(
                        pp, f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")
            gt_files.append(label_path)

        return gt_files


    @staticmethod
    def findPredictionFiles(path):
        return [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith(".nii.gz")]

    @staticmethod
    def finalArrangements(predFolder: str, origFolder: str):
        """
        """
        raise Exception("deprecated")
