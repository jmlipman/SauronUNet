from tensorboardX import SummaryWriter
import torch, os, time, json, inspect, re, pickle
from datetime import datetime
import numpy as np
import torchio as tio
#from IPython import embed
from lib.metric import Metric
from lib.utils import he_normal
from typing import List, Callable, Type, Tuple
from lib.data.BaseDataset import BaseDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torchio.data.dataset import SubjectsDataset
from torchio.data.subject import Subject
import nibabel as nib


def callCallbacks(callbacks: List[Callable], prefix: str,
        allvars: dict) -> None:
    """
    Call all callback functions starting with a given prefix.
    Check which inputs the callbacks need, and, from `allvars` (that contains
    locals()) pass those inputs.
    Read more about callback functions in lib.utils.callbacks

    Args:
      `callbacks`: List of callback functions.
      `prefix`: Prefix of the functions to be called.
      `allvars`: locals() containing all variables in memory.
    """
    for c in callbacks:
        if c.__name__.startswith(prefix):
            input_params = inspect.getfullargspec(c).args
            required_params = {k: allvars[k] for k in input_params}
            c(**required_params)


def unwrap_data(subjects_batch: dict, data: Type[BaseDataset],
        device: str) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Extract X (input data), Y (labels), and W (weights, optional) from
    subects_batch, which comes from the tr/val_loader.

    Args:
      `subjects_batch`: Contains the train/val/test data.
      `data`: Dataset. For extracting the dimension and infoXYW.
      `device`: Device where the computations will be performed ("cuda").

    Returns:
      `: Input data.
      `Y`: Labels.
      `W`: (maybe empty) Weights.
    """

    X, Y, W = [], [], []
    for c, infoN in zip([X, Y, W], [data.infoX, data.infoY, data.infoW]):
        for td in infoN:
            # If the data class contains test data without 'label',
            # then, `td` is not in `subjects_batch`.
            if td in subjects_batch:
                c.append( subjects_batch[td][tio.DATA].to(device) )
            if data.dim == "2D" and len(c) > 0:
                c[-1] = c[-1].squeeze(dim=-1)

    return (X, Y, W)

class BaseModel(torch.nn.Module):
    """
    Models inherit this class, allowing them to perform training and
    evaluation.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def initialize(self, device: str, model_state: str, log, isSauron=False) -> None:
        """
        Initializes the model.
        Moves the operations to the selected `device`, and loads the model's
        parameters or initializes the weights/biases.

        Args:
          `device`: Device where the computations will be performed.
          `model_state`: Path to the parameters to load, or "".
          `log` (lib.utils.handlers.Log).
          `isSauron`: whether we are loading a model for Sauron. It requires
           changing a bit how to load the weights.
        """
        self.device = device
        self.to(self.device)

        # Load or initialize weights
        if model_state != "":
            log("Loading previous model")
            params = torch.load(model_state)
            if isSauron:
                params_k = list(params)
                for k in params_k:
                    newk = k.split(".")
                    newk.insert(0, "network")
                    newk.insert(3, "module")
                    newk = ".".join(newk)
                    params[newk] = params[k]
                    del params[k]

            #from IPython import embed; embed()
            #self.load_state_dict(torch.load(model_state))
            self.load_state_dict(params)
        else:
            def weight_init(m):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                    he_normal(m.weight)
                    if m.bias is not None: # for convs with bias=False
                        torch.nn.init.zeros_(m.bias)
            self.apply(weight_init)

        param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log("Number of parameters: " + str(param_num))

    def fit(self, tr_loader: DataLoader, val_data: SubjectsDataset,
            epoch_start: int,
            epochs: int, val_interval: int, loss: Callable,
            val_batch_size: int,
            opt: Type[Optimizer],
            scheduler: Type[_LRScheduler],
            callbacks: List[Callable], log,
            history: dict) -> None:
        """
        Trains the NN.
        Note: I was wondering why the values in scores/results-LastEpoch.json
        and the masks in preds/ do not match (although they strongly correlate).
        The reason is that there is a pruning step in between these two.
        Therefore, "it's not a bug, it's a feature", i.e., my analysis are valid.

        Args:
          `tr_loader`: Training data.
          `val_loader`: Validaiton set.
          `epoch_start`: Epoch in which the training will start (usually 1).
          `epochs`: Epochs to train the model. If 0, no train.
          `val_interval`: After how many epochs to perform validation.
          `loss`: Loss function.
          `val_batch_size`: Batch size at validation time.
          `opt`: Optimizer.
          `scheduler`: Learning rate scheduler (for lr decay).
          `callbacks`: List of callback functions.
          `log` (lib.utils.handlers.Log).
        """
        t0 = time.time()
        outputPath = log.path.replace("log.txt", "")
        e = 1

        # Used by callback functions (Sauron)
        channels_history = history["channels_history"]
        val_loss_history = history["val_loss_history"]
        tr_loss_history = history["tr_loss_history"]
        if "mod_patience" in history:
            from lib.models.Sauron import DropChannels
            for mod in self.modules():
                if isinstance(mod, DropChannels):
                    mod.thr = history["mod_thr"][mod.name]
                    mod.patience = history["mod_patience"][mod.name]

        if len(val_data) > 0:
            os.makedirs(f"{outputPath}/scores")

        # Tensoboard path
        tb_path = "/".join(outputPath.split("/")[:-3]) + "/tensorboard/" + "_".join(outputPath.split("/")[-3:])[:-1]
        writer = SummaryWriter(tb_path)

        # As some servers only allow you to run jobs for max. 3 days, it
        # can be important to resume the training and re-execute certain
        # procedures, like scheduler.step()
        if epoch_start > 1:
            for e in range(1, epoch_start):
                #callCallbacks(callbacks, "_start_epoch", locals())
                #callCallbacks(callbacks, "_end_epoch", locals())
                if scheduler:
                    scheduler.step()
            e += 1

        callCallbacks(callbacks, "_start_training", locals())

        while e <= epochs:

            callCallbacks(callbacks, "_start_epoch", locals())

            tr_loss = self.fit_oneepoch(tr_loader=tr_loader, loss=loss,
                    opt=opt, callbacks=callbacks)
            writer.add_scalar("tr_loss", tr_loss, e)
            tr_loss_history.append(tr_loss)

            val_str = ""
            if len(val_data) > 0 and e % val_interval == 0:
                log("Validation")
                val_str = self.evaluate(val_data, val_batch_size,
                        f"{outputPath}/scores/results-{e}.json",
                        loss, callbacks)
                val_loss = float(re.match("Val loss: ([0-9]*\.[0-9]*)", val_str)
                                .group().split(" ")[-1])
                val_loss_history.append(val_loss)
                writer.add_scalar("val_loss", val_loss, e)

            callCallbacks(callbacks, "_end_epoch", locals())
            if hasattr(self, "end_training") and self.end_training:
                break

            if scheduler:
                scheduler.step()

            eta = datetime.fromtimestamp(time.time() + (epochs-e)*(time.time()-t0)/e).strftime("%Y-%m-%d %H:%M:%S")
            log(f"Epoch: {e}. Loss: {tr_loss}. {val_str} ETA: {eta}")
            e += 1
            writer.close()

            # Save utilized GPU memory
            #out = os.popen('nvidia-smi').read()
            #with open("output_nvidia-smi", "w") as f:
            #    f.write(out)
            #raise Exception("para")
            #if e > 400: # For stopping the training before 3 days (kits)
            #    break
        callCallbacks(callbacks, "_end_training", locals())



    def fit_oneepoch(self, tr_loader: DataLoader, loss: Callable,
            opt: Type[Optimizer], callbacks: List[Callable]) -> float:
        """
        Train the model for a single epoch.

        Args:
          `tr_loader`: Training data.
          `loss`: Loss function.
          `opt`: Optimizer.
          `callbacks`: List of callback functions.

        Returns:
          Training loss.
        """
        self.train()
        tr_loss = 0

        for tr_i, subjects_batch in enumerate(tr_loader):
            X, Y, W = unwrap_data(subjects_batch,
                    tr_loader.dataset.dataset, self.device)
            info = subjects_batch["info"]

            callCallbacks(callbacks, "_start_train_iteration", locals())

            # Here I could add a param to indicate whether it's the last
            # then, in such case, compute the distance for the thr
            output = self(X)
            tr_loss_tmp = loss(output, Y)
            tr_loss += tr_loss_tmp.cpu().detach().numpy()

            # Optimization
            opt.zero_grad()
            tr_loss_tmp.backward()
            callCallbacks(callbacks, "_after_compute_grads", locals())
            opt.step()

            callCallbacks(callbacks, "_end_train_iteration", locals())
            #raise Exception("para epoch")

        tr_loss /= len(tr_loader)

        return tr_loss

    def predict(self, data: SubjectsDataset, batch_size: int,
            path_output: str, path_original: str="") -> None:
        """
        Generate and save the images.

        Args:
          `data`: Input images.
          `batch_size`: Batch size.
          `path_output`: Folder where the images will be saved.
        """
        self.eval()
        with torch.no_grad():
            for sub_i, subject in enumerate(data):
                print(f"Prediction: {sub_i+1}/{len(data)}")
                y_pred, _ = self.predict_subject(subject, batch_size,
                                                 data.dataset)
                y_pred_cpu = y_pred.cpu().detach().numpy()

                orig_im = nib.load(subject.info["path"])
                data.dataset.save(
                        y_pred_cpu,
                        affine=orig_im.affine,
                        header=orig_im.header,
                        path_output=os.path.join(path_output,
                                            f"{subject.info['id']}.nii.gz"))

                # Postprocess
                if path_original != "":
                    im = data.dataset.post_process(y_pred_cpu, subject,
                            path_original)

                    # Save
                    data.dataset.save(
                            im.get_fdata(),
                            affine=im.affine,
                            header=im.header,
                            path_output=os.path.join(path_output+"_post",
                                                f"{subject.info['id']}.nii.gz"))


    def evaluate(self, data: SubjectsDataset, batch_size: int,
            path_scores: str, loss: Callable=None,
            callbacks: List[Callable]=[]) -> str:
        """
        Evaluate the data with their labels. Optionally, save output.

        Args:
          `data`: Data to be evaluated.
          `batch_size`: Batch size.
          `path_scores`: Filepath where metrics will be saved.
          `loss`: Optional loss function.
          `path_out`: Folder path where output images will be saved.

        Returns:
          `val_str`: Validation errors in text form.
        """
        self.eval()

        metrics = ["dice", "HD", "TFPN"]
        Measure = Metric(metrics, onehot=data.dataset.onehot,
                classes=data.dataset.classes,
                classes_mean=data.dataset.measure_classes_mean,
                multiprocess=True)
        results = {}

        val_loss = 0
        with torch.no_grad():
            for sub_i, subject in enumerate(data):
                # NOTE: Does this work the same way if I give entire 3D images?
                #       In that case, grid_sampler and aggregator are useless.
                callCallbacks(callbacks, "_start_val_subject", locals())

                y_pred, val_loss_tmp = self.predict_subject(subject,
                        batch_size, data.dataset, loss)
                y_pred_cpu = y_pred.cpu().numpy()
                y_true_cpu = subject.label.numpy()

                results[subject["info"]["id"]] = Measure.all(y_pred_cpu,
                        y_true_cpu, subject["info"])

                if val_loss_tmp != -1:
                    val_loss += val_loss_tmp * (1 / len(data))

                callCallbacks(callbacks, "_end_val_subject", locals())

        # Synchronize
        if Measure.multiprocess:
            for k in results:
                results[k] = results[k].get()
            # If we are using multiprocessing we need to close the pool
            Measure.close()

        with open(path_scores, "w") as f:
            f.write(json.dumps(results))

        val_str = ""
        if not loss is None:
            val_str += f"Val loss: {val_loss}. "
        #val_str += Measure.getMeanValScores(results)
        val_str += Measure.getAllValScores(results)

        return val_str

    def predict_subject(self, subject: Subject, batch_size: int,
            dataset: Type[BaseDataset],
            loss: Callable=None) -> Tuple[Tensor, float]:
        """
        Prediction for `subject`. Called at validation or test time.
        NOTE: Check this is fine when subjects are whole-images.

        Args:
          `subject`: Single subject from which slices will be extracted
                     according to its `patch_size`.
          `batch_size`: Batch size.
          `dataset`: For unwrapping the data.
          `loss`: Optional loss function.

        Returns:
          `y_pred`: Predicted subject.
          `val_loss`: Validation loss (or -1 if no `loss` was given).
        """
        # If patch size is larger than the actual image (as in ACDC17)
        # this will calculate the amount of padding needed by the grid sampler.
        sub_size = np.array(subject[dataset.infoX[0]][tio.DATA].shape[1:])
        patch_size = np.array(subject.info["patch_size"])
        diff = sub_size - patch_size
        overlap = diff*(diff<0)*-1 # There could be odd dimensions
        overlap[(overlap!=0) & (overlap%2!=0)] += 1

        grid_sampler = tio.inference.GridSampler(subject,
                patch_size=subject.info["patch_size"],
                patch_overlap=overlap, padding_mode="constant")
        aggregator = tio.inference.GridAggregator(grid_sampler)
        subject_loader = torch.utils.data.DataLoader(grid_sampler,
                batch_size=batch_size)

        val_loss = -1 if loss is None else 0

        for val_i, subjects_batch in enumerate(subject_loader):

            X, Y, W = unwrap_data(subjects_batch,
                    dataset, self.device)

            locations = subjects_batch[tio.LOCATION]
            info = subjects_batch["info"]
            id_ = info["id"][0]

            output = self(X)

            if loss is not None:
                val_loss_tmp = loss(output, Y)
                val_loss += val_loss_tmp.cpu().detach().numpy()
            else:
                # To enhance the output, the input image is flipped and the
                # output is averaged. 4 rotations for 2D images, and 8 for 3D.
                # Note that for this to make sense, it has to match the D.A.
                if len(output[0].shape) == 4:
                    flips = [(3, ), (2, ), (3, 2)]
                elif len(output[0].shape) == 5:
                    flips = [(4, ), (3, ), (4, 3), (2, ), (4, 2), (3, 2), (4, 3, 2)]
                else:
                    raise Exception("Not sure why, but the length of the shape "
                            "of the output is different than expected: "
                            f"{output[0].shape}")

                output[0] /= len(flips)+1
                for flip in flips:
                    tmp_out = self([X[0].flip(flip)])
                    output[0] += (tmp_out[0].flip(flip) / (len(flips)+1))

            if "Sauron" in str(type(self)):
                output = output[0]

            # If images were originally 3D and we use 2D patches, expand dim
            if (len(info["patch_size"]) == 3 and
                    len(output[0].shape) == 4): # make it output[0][0] when Sauron
                aggregator.add_batch(torch.unsqueeze(output[0], -1), locations)
            else:
                aggregator.add_batch(output[0], locations)
        y_pred = aggregator.get_output_tensor()

        if loss is not None:
            val_loss /= len(subject_loader)

        return (y_pred, val_loss)
