import torch, os, random, time, re
import nibabel as nib
import numpy as np
from lib.data.BaseDataset import BaseDataset
from lib.loss import DS_CrossEntropyDiceLoss, DS_CrossEntropyDiceLoss_Distance
from typing import List
import torchio as tio
from lib.models.UNet import UNet
from lib.models.nnUNet import nnUNet
from lib.models.Sauron import Sauron
from torch.nn import InstanceNorm2d, BatchNorm2d
from torch.optim.lr_scheduler import LambdaLR
import lib.utils as utils

class ACDC17Dataset(BaseDataset):
    """
    ACDC 2017 Challenge.
    Paper: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
    "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures
    Segmentation and Diagnosis: Is the Problem Solved?", in TMI (2018).

    """
    name = "acdc17"
    modalities = 1
    classes = {0: "background", 1: "RV_cavity", 2: "myocardium", 3: "LV_cavity"}
    dim = "2D"
    # Specify how the final prob. values are computed: softmax or sigmoid?
    onehot = "softmax"
    #onehot = lambda x: x
    # Which classes will be reported during validation
    measure_classes_mean = np.array([1, 2, 3])
    # These values are filled by get_dataset() in lib/helpers.py
    #data_path = "/media/miguelv/HD1/Datasets/ACDC17/"
    #data_path = "/home/miguelv/data/in/ACDC17/"
    infoX = ["im"]
    infoY = ["label"]
    infoW = []

    # Sauron properties. Leave here for logging.
    dist_fun = "euc_norm"
    imp_fun = "euc_rand"
    #dist_fun = ""
    #imp_fun = ""
    sf = 2

    # Optimization strategy
    opt = {"architecture":# levels=7, fms_init=48 original
                {"modalities": modalities, "n_classes": len(classes),
                    "dist_fun": dist_fun, "imp_fun": imp_fun, "sf": sf,
                    "fms_init": 48, "levels": 7, "normLayer": BatchNorm2d,
                    "dim": "2D"},
                "loss": DS_CrossEntropyDiceLoss_Distance,
            "batch_size": 10,
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
            tio.RandomFlip(axes=(0,), flip_probability=0.5), #axis=0
            tio.RandomFlip(axes=(1,), flip_probability=0.5), #axis=1
            # control points = 14 is not bad neither
            tio.transforms.RandomElasticDeformation(num_control_points=7,
                locked_borders=2, p=0.3),
            tio.CropOrPad((320, 320, 0)), # (centered of the image)
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])
    transforms_val = tio.Compose([
            tio.CropOrPad((320, 320, 0)),
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])
    transforms_test = tio.Compose([
            #tio.CropOrPad((320, 320, 0)),
            tio.transforms.OneHot(num_classes=len(classes)),
            tio.transforms.ZNormalization(),
        ])

    def __init__(self, split: List[float], seed: int):
        """Prepares the data into self.list

           Args:
            `split`: Percentage of data used for training and validation.
        """
        self.transforms_dict = {"train": self.transforms_train,
                "validation": self.transforms_val,
                "test": self.transforms_test}


        print("Loading data...")
        t0 = time.time()

        ids = np.arange(1, 101) # Patients ids are from 1 to 100
        random.seed(seed)
        random.shuffle(ids)

        # Test is 20% of the training data since I cannot use the online platform
        test_ids = ids[:int(len(ids)*0.2)]
        ids = ids[int(len(ids)*0.2):]
        tr_ids = ids[:int(len(ids)*split[0])]
        val_ids = ids[int(len(ids)*split[0]):]
        #test_ids = np.arange(101, 151) # 151

        subfolder_test = "training_preprocess"
        subfolder_train = "training_preprocess"

        #test_ids = ids[0:1]
        #tr_ids = ids[0:1]
        #val_ids = ids[0:1]

        self.subjects_dict = {"train": [], "validation": [], "test": []}
        for ids, partition in zip([tr_ids, val_ids], ["train", "validation"]):
            for im_id in ids:
                patient_path = os.path.join(
                        self.data_path, subfolder_train,
                        "patient" + str(im_id).zfill(3))

                # For each patient, there are 2 segmented 3D images
                images_path = [os.path.join(patient_path, p) for p in os.listdir(patient_path) if not "gt" in p]
                #print("Only one image part 2")
                for image_path in images_path:
                    tmp_im = nib.load(image_path)
                    spacing = np.array(tmp_im.header.get_zooms())
                    slices = tmp_im.shape[-1]

                    self.subjects_dict[partition].append(tio.Subject(
                        im=tio.ScalarImage(image_path),
                        label=tio.LabelMap(image_path.replace(".nii", "_gt.nii")),
                        info={
                            "voxelspacing": spacing,
                            "id": image_path.split("/")[-1].replace(".nii.gz", ""),
                            "path": image_path,
                            "slices": slices,
                            "patch_size": (320, 320, 1)
                            }
                        ))

        # Reading test images (if any)
        for im_id in test_ids:
            patient_path = os.path.join(self.data_path, subfolder_test,
                    "patient" + str(im_id).zfill(3))

            # For each patient, there are 2 segmented 3D images
            images_path = [os.path.join(patient_path, p)
                    for p in os.listdir(patient_path) if not p.endswith("_gt.nii.gz")]

            for image_path in images_path:
                tmp_im = nib.load(image_path)
                spacing = np.array(tmp_im.header.get_zooms())
                slices = tmp_im.shape[-1]

                self.subjects_dict["test"].append(tio.Subject(
                    im=tio.ScalarImage(image_path),
                    info={
                        "voxelspacing": spacing,
                        "id": image_path.split("/")[-1].replace(".nii.gz", ""),
                        "path": image_path,
                        "slices": slices,
                        #"patch_size": (tmp_im.shape[0], tmp_im.shape[1], 1)
                        "patch_size": (320, 320, 1)
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
        Verifies that the required data is in the expected location.

        Args:
          `inputFolder`: Folder where 'training' folder (from ACDC17 challenge)
                         is located.

        Returns:
          Whether the expected data is in the right place.
        """

        def _verify_folders_and_files(path, expected_folders):
            verify_ok = True
            for folder in expected_folders:
                files = os.listdir(os.path.join(path, folder))
                images, gts = [], []
                for f in files:
                    if len(re.findall("patient[01][0-9][0-9]_frame[01][0-9]\.nii\.gz", f)) == 1:
                        images.append(f)
                    elif len(re.findall("patient[01][0-9][0-9]_frame[01][0-9]_gt\.nii\.gz", f)) == 1:
                        gts.append(f)

                if len(expected_folders) == 100: # Training
                    if len(images) != 2 or len(gts) != 2:
                        print(f"> Warning in `{folder}`: Expected data (2 images and 2 GTs) missing.")
                        verify_ok = False
                elif len(expected_folders) == 50: # Test
                    if len(images) != 2:
                        print(f"> Warning in `{folder}`: Expected data (2 images) missing.")
                        verify_ok = False
                else:
                    verify_ok = False

            return verify_ok


        # Expected folders
        exp_folders_train = set(["patient" + str(x).zfill(3) for x in range(1, 101)])
        exp_folders_test = set(["patient" + str(x).zfill(3) for x in range(101, 151)])

        found_folders = set(os.listdir(inputFolder))

        diff1 = exp_folders_train - found_folders
        diff2 = exp_folders_test - found_folders

        if len(diff1) == 0:
            # Input folder contains the training images
            return _verify_folders_and_files(inputFolder, list(exp_folders_train))
        elif len(diff2) == 0:
            # Input folder contains the test images
            return _verify_folders_and_files(inputFolder, list(exp_folders_test))
        else:
            raise Exception(f"--input folder `{inputFolder}` does not contain all"
                    " required subfolders (patient[1-100] for training and"
                    " patient[101-150] for testing)")

    @staticmethod
    def pre_process(inputFolder: str, outputFolder: str, _=None) -> None:
        """
        Preprocesses the segmentations.

        Args:
          `inputFolder`: Folder where the predictions are located.
          `outputFolder`: Folder where the postprocessed predictions will be saved.
          `_`: Unnecessary parameter added for the compatibility with post_process.
        """

        folders = os.listdir(inputFolder)
        subjects = []

        print("Preprocessing files...")
        for i, folder in enumerate(folders):
            os.makedirs(os.path.join(outputFolder, folder))
            files = os.listdir(os.path.join(inputFolder, folder))
            for file in files:
                if len(re.findall("patient[01][0-9][0-9]_frame[01][0-9]\.nii\.gz", file)) == 1:
                    fullpath = os.path.join(inputFolder, folder, file)
                    fullpath_gt = fullpath.replace(".nii.gz", "_gt.nii.gz")

                    # Resample image (and gt)
                    image = nib.load(fullpath)
                    voxres = list(image.header.get_zooms())
                    voxres.append(1)
                    voxres = np.array(voxres)
                    images = [tio.ScalarImage(
                                tensor=image.get_fdata()[np.newaxis, ...],
                                affine=voxres*image.affine)]

                    if os.path.isfile(fullpath_gt):
                        gt = nib.load(fullpath_gt)
                        images.append( tio.LabelMap(
                                    tensor=gt.get_fdata()[np.newaxis, ...],
                                    affine=voxres*gt.affine) )
                        image, gt = utils.resamplev2(images=images,
                                      voxres=(1.25, 1.25, voxres[2]))

                    else:
                        image = utils.resamplev2(images=images,
                                      voxres=(1.25, 1.25, voxres[2]))

                    image_data = image.get_fdata()
                    image_data = (image_data - image_data.mean()) / image_data.std()
                    image = nib.Nifti1Image(image_data, affine=image.affine,
                            header=image.header)

                    # Change its data type
                    image.set_data_dtype(np.dtype("float32"))
                    nib.save(image, os.path.join(outputFolder, folder, file))
                    if os.path.isfile(fullpath_gt):
                        gt.set_data_dtype(np.dtype("float32"))
                        nib.save(gt, os.path.join(outputFolder, folder,
                                   file.replace(".nii.gz", "_gt.nii.gz")))


    @staticmethod
    def post_process(image: np.array, subject: tio.data.subject.Subject,
            path_original: str) -> nib.Nifti1Image:
        """
        Postprocess the data.

        Args:
          `image`: Image (CHDW).
          `subject`: Subject `image` belongs to.
          `path_original`: Path where all the images can be found.

        Return:
          Post-processed image.
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

    @staticmethod
    def findGroundTruthFiles(path, predFiles):
        gts = []
        for p in predFiles:
            filename = p.split("/")[-1]
            patient = filename.split("_")[0]
            gts.append(os.path.join(path, patient, filename.replace(".nii.gz", "_gt.nii.gz")))
        return gts

    @staticmethod
    def findPredictionFiles(path):
        return [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith(".nii.gz") ]

    @staticmethod
    def finalArrangements(predFolder: str, origFolder: str) -> None:
        """
        Arranges the post-processed data. Useful to rename and compress the
        predictions to prepare them for the submission.

        Args:
          `predFolder`: Folder where all predictions are located.
          `origFolder`: Folder where the original data are located.
        """

        """
        folders = os.listdir(origFolder)
        esed_dict = {}
        from IPython import embed; embed()
        for folder in folders:
            frames = [x for x in os.listdir(f"{origFolder}/{folder}") if "frame" in x]
            print(folder)
            with open(f"{origFolder}/{folder}/Info.cfg", "r") as f:
                d = f.read()
                ed = re.findall("ED: ([0-9]+)", d)[0].zfill(2)
                es = re.findall("ES: ([0-9]+)", d)[0].zfill(2)
                esed_dict[f"{folder}_frame{ed}.nii.gz"] = f"{folder}_ED.nii.gz"
                esed_dict[f"{folder}_frame{es}.nii.gz"] = f"{folder}_ES.nii.gz"

        for f in os.listdir(predFolder):
            p1 = os.path.join(predFolder, f)
            p2 = os.path.join(predFolder, esed_dict[f])
            os.rename(p1, p2)

        os.system(f"cd {predFolder} && zip patients.zip patient*.nii.gz")
        """
        pass # Unnecessary
