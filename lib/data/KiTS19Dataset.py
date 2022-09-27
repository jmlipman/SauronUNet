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

class KiTS19Dataset(BaseDataset):
    """
    KiTS 2019 Challenge.
    Paper: https://arxiv.org/abs/1912.01054
    """
    name = "kits19"
    modalities = 1
    classes = {0: "background", 1: "kidney", 2: "tumor"}
    dim = "3D"
    # Specify how the final prob. values are computed: softmax or sigmoid?
    onehot = "softmax"
    # Which classes will be reported during validation
    measure_classes_mean = np.array([1, 2])
    # These values are filled by get_dataset() in lib/helpers.py
    #data_path = "/media/miguelv/HD1/Datasets/KiTS19/" # train/test_preprocess
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
            "batch_size": 2,
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

        exclude = [15, 23, 37, 68, 125, 133]
        ids = ["case_" + str(x).zfill(5) for x in range(210) if x not in exclude]
        random.seed(seed)
        random.shuffle(ids)

        tr_ids = ids[:int(len(ids)*split[0])]
        val_ids = ids[int(len(ids)*split[0]):]
        test_ids = ["case_" + str(x).zfill(5) for x in range(210, 300)]

        # Test overfitting
        #tr_ids = ids[0:1]
        #val_ids = ids[0:1]
        #test_ids = test_ids[0:1]
        #tr_ids = ["case_00113"]
        #val_ids = ["case_00113"]
        #test_ids = ["case_00113"]

        self.subjects_dict = {"train": [], "validation": [], "test": []}
        for ids, partition in zip([tr_ids, val_ids], ["train", "validation"]):
            for im_id in ids:
                image_path = os.path.join(
                        self.data_path, "train_preprocess", im_id, "imaging.nii.gz")
                label_path = os.path.join(
                        self.data_path, "train_preprocess", im_id, "segmentation.nii.gz")

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
                        "patch_size": (80, 160, 160)
                        }
                    ))

        for im_id in test_ids:
            image_path = os.path.join(
                    self.data_path, "test_preprocess", im_id, "imaging.nii.gz")

            tmp_im = nib.load(image_path)
            spacing = np.array(tmp_im.header.get_zooms())
            slices = 2

            self.subjects_dict["test"].append(tio.Subject(
                im=tio.ScalarImage(image_path),
                info={
                    "voxelspacing": spacing,
                    "id": im_id,
                    "path": image_path,
                    "slices": slices,
                    "patch_size": (80, 160, 160)
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
          `inputFolder`: Folder where the training data is located.
                         case_0000 - case_00209

        Returns:
          Whether the expected data is in the right place.
        """

        # Expected folders
        exp_folders_tr = set(["case_" + str(x).zfill(5) for x in range(210)])
        exp_folders_te = set(["case_" + str(x).zfill(5) for x in range(210, 300)])
        found_folders = set(os.listdir(inputFolder))
        found_folders.discard("kits.json") # In case these exist
        found_folders.discard("LICENSE")

        if len(exp_folders_tr - found_folders) > 0:
            if len(exp_folders_te - found_folders) > 0:
                raise Exception(f"For the training, the following files are"
                        "expected in folder `{inputFolder}`:"
                        "'case_00000', ..., 'case_00209'; for testing: "
                        "'case_00210', ..., 'case_00299'")
            else:
                # Test folder found
                for folder in ["case_" + str(x).zfill(5) for x in range(210, 300)]:
                    tmp_folder = os.path.join(inputFolder, folder)
                    files = os.listdir(tmp_folder)
                    if (len(files) != 1 or
                        len(set(files)-set(["imaging.nii.gz"])) > 0):
                        raise Exception(f"One file expected in `{tmp_folder}`: "
                                "'imaging.nii.gz'")
        else:
            # Train folder found
            for folder in ["case_" + str(x).zfill(5) for x in range(210)]:
                tmp_folder = os.path.join(inputFolder, folder)
                files = os.listdir(tmp_folder)
                if (len(files) != 2 or
                    len(set(files)-set(["imaging.nii.gz", "segmentation.nii.gz"])) > 0):
                    raise Exception(f"Two files expected in `{tmp_folder}`: "
                            "'imaging.nii.gz' and 'segmentation.nii.gz'")

        return True

    @staticmethod
    def pre_process(inputFolder: str, outputFolder: str, _=None) -> None:
        """
        Preprocesses the segmentations.

        Args:
          `inputFolder`: Folder where the predictions are located.
          `outputFolder`: Folder where the postprocessed predictions will be saved.
          `_`: Unnecessary parameter added for the compatibility with post_process.
        """

        if "case_00001" in os.listdir(inputFolder):
            # Train
            exclude = [15, 23, 37, 68, 125, 133]
            folders = ["case_" + str(x).zfill(5) for x in range(210) if x not in exclude]
        else:
            # Test
            folders = ["case_" + str(x).zfill(5) for x in range(210, 300)]

        subjects = []
        print("Processing files...")
        for i, folder in enumerate(folders):
            print(f"Case: {i+1}/{len(folders)}")
            os.makedirs(os.path.join(outputFolder, folder))
            fullpath = os.path.join(inputFolder, folder, "imaging.nii.gz")
            fullpath_gt = os.path.join(inputFolder, folder, "segmentation.nii.gz")

            if int(folder.replace("case_", "")) < 210:
                image = nib.load(fullpath)
                gt = nib.load(fullpath_gt)
                # Important line below. For some reason, some GTs have different
                # sform than their images. This is weird because at this point I'm
                # dealing with the provided data. This is important because if
                # they are different torchio will complain when loading the subjects
                #
                # "self.check_consistent_attribute('direction')"
                # file: torchio/data/subject.py
                gt.set_sform(image.get_sform()) # Important
                images = [tio.ScalarImage(tensor=image.get_fdata()[np.newaxis, ...],
                                          affine=image.affine),
                          tio.LabelMap(tensor=gt.get_fdata()[np.newaxis, ...],
                                       affine=gt.affine)]
                image, gt = utils.resamplev2(images=images,
                                             voxres=(3.22, 1.62, 1.62))

            else:
                image = nib.load(fullpath)
                images = [tio.ScalarImage(tensor=image.get_fdata()[np.newaxis, ...],
                                          affine=image.affine)]
                image = utils.resamplev2(images=images,
                                         voxres=(3.22, 1.62, 1.62))

            image_data = np.clip(image.get_fdata(), -79, 304)
            image_data = (image_data - 101) / 76.9
            image_data = image_data.astype(np.float32)

            # Pad images smaller than the patch size: 80x160x160.
            padding = []
            for mmin, curr_size in zip([80, 160, 160], image_data.shape):
                if curr_size >= mmin:
                    padding.append((0, 0))
                else:
                    ini = (mmin - curr_size) // 2
                    fin = mmin - curr_size - ini
                    padding.append((ini, fin))

            image_data = np.pad(image_data, padding)
            image = nib.Nifti1Image(image_data, affine=image.affine,
                    header=image.header)
            image.set_data_dtype(np.float32)
            nib.save(image, os.path.join(outputFolder, folder, "imaging.nii.gz"))
            # I write this padding so that I know later what pixels to unpad
            # from the prediction
            with open(os.path.join(outputFolder, "padding"), "a") as f:
                f.write(f"{folder}*{str(padding)}\n")

            if int(folder.replace("case_", "")) < 210:
                gt_data = gt.get_fdata()
                gt_data = np.pad(gt_data, padding)
                gt = nib.Nifti1Image(gt_data, affine=gt.affine, header=gt.header)
                gt.set_data_dtype(np.float32)
                nib.save(gt, os.path.join(outputFolder, folder, "segmentation.nii.gz"))

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
        padding_file = os.path.join("/".join(subject["info"]["path"].split("/")[:-2]),
                                    "padding")
        with open(padding_file, "r") as f:
            padding_tmp = f.read().split("\n")[:-1]
        pads = {}
        for pad_line in padding_tmp:
            case, pad_tmp = pad_line.split("*")
            pads[case] = eval(pad_tmp)

        # Example: curr_case = case_00210
        curr_case = subject["info"]["path"].split("/")[-2]

        # First, unpad
        ps = np.array(image.shape)[1:]
        p = np.array(pads[curr_case])
        ix, jx = p[0,0]+0, ps[0]-p[0,1]
        iy, jy = p[1,0]+0, ps[1]-p[1,1]
        iz, jz = p[2,0]+0, ps[2]-p[2,1]

        unpadded_im = image[:, ix:jx, iy:jy, iz:jz]

        aff = nib.load(subject["info"]["path"]).affine
        images = [tio.ScalarImage(tensor=unpadded_im,
                                  affine=aff)]
        orig_im = nib.load(os.path.join(path_original, curr_case, "imaging.nii.gz"))
        image = utils.resamplev2(images=images, size=orig_im.shape)

        return image

    @staticmethod
    def post_process_old(inputFolder: str, outputFolder: str,
                     original: str) -> None:
        """
        Postprocesses the segmentations.

        Args:
          `inputFolder`: Folder where the predictions are located.
          `outputFolder`: Folder where the postprocessed predictions will be saved.
          `original`: Folder containing the original files.
        """
        raise Exception("deprecated")
        folders = [f"case_{str(x).zfill(5)}" for x in range(210, 300)]
        padding_file = "/media/miguelv/HD1/Datasets/KiTS19/test_preprocess/padding"
        with open(padding_file, "r") as f:
            padding_tmp = f.read().split("\n")[:-1]
        pads = {}
        for pad_line in padding_tmp:
            case, pad_tmp = pad_line.split("*")
            pads[case] = eval(pad_tmp)

        for i, folder in enumerate(folders[:1]):
            print(f"Postprocessing patient {i+210}/300")

            tmp_path = os.path.join(original, folder, "imaging.nii.gz")
            im = nib.load(tmp_path)
            pred_path = os.path.join(inputFolder, f"{folder}.nii.gz")
            # I'm resampling to the original size, but my prediction actually
            # contains some padded values... First I have to remove these values
            # Or maybe I can resample to a voxel resolution (the original one)
            # and then remove the padded values, IDK.

            # First, load, unpad, save
            pred = nib.load(pred_path)
            ps = np.array(pred.shape)

            p = np.array(pads[folder])
            ix, jx = p[0,0]+0, ps[0]-p[0,1]
            iy, jy = p[1,0]+0, ps[1]-p[1,1]
            iz, jz = p[2,0]+0, ps[2]-p[2,1]

            pred_data = pred.get_fdata()[ix:jx, iy:jy, iz:jz]
            output_path =  os.path.join(outputFolder, f"{folder}.nii.gz")
            pred_im = nib.Nifti1Image(pred_data, affine=pred.affine,
                    header=pred.header)
            nib.save(pred_im, output_path)

            #from IPython import embed; embed()
            #raise Exception("PROBLEM HERE")

            pred = resample(image_path=output_path, size=im.shape)
            la = pred.get_fdata()
            p = la[12, 304:315, 193:203]
            q = pred_data[19, 163:169, 104:108]
            print(np.round(p, 3))
            print(np.round(q, 3))
            from IPython import embed; embed()
            #print(f"Original: {im.get_fdata().shape}. Prediction: {pred.get_fdata().shape}")
            res_im = nib.Nifti1Image(la, affine=im.affine, header=im.header)

            #from IPython import embed; embed()
            # Without the following line, predictions become quite heavy
            # and the online platform will throw an error:
            # "The container was killed as it exceeded the memory limit of 4g."
            res_im.header.set_data_dtype(np.dtype("uint8"))
            output_path =  os.path.join(outputFolder,
                    f"{folder}.nii.gz".replace("case", "prediction"))
            nib.save(res_im, output_path)

            # For some reason, this won't save the labels correctly
            # Instead of [0, 1, 2] -> [0, 0.9999, 2.00001]
            # So, we do the following:
            tt = nib.load(output_path)
            newim = nib.Nifti1Image(nib.casting.float_to_int(tt.get_fdata(),
                                                 int_type=np.dtype("uint8")),
                                    affine=tt.affine, header=tt.header)
            nib.save(newim, output_path)

    @staticmethod
    def findGroundTruthFiles(path):
        #raise NotImplementedError("para")
        return [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith(".nii.gz")]

    @staticmethod
    def findPredictionFiles(path):
        #raise NotImplementedError("para")
        return [os.path.join(path, x) for x in sorted(os.listdir(path)) if x.endswith(".nii.gz")]

    @staticmethod
    def finalArrangements(predFolder: str, origFolder: str):
        """
        Arranges the post-processed data. Useful to rename and compress the
        predictions to prepare them for the submission.

        Args:
          `predFolder`: Folder where all predictions are located.
          `origFolder`: Folder where the original data are located.
        """
        os.system(f"rename 's/case/prediction/g' {predFolder}/case*")
        os.system(f"cd {predFolder} && zip predictions.zip prediction_*.nii.gz")
