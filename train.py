###########################################
# This script trains great UNet baselines #
# Prior to this, run preprocess.py #
###########################################

import torch, os, time
from lib.utils import parseArguments, Log
from torch.utils.data import DataLoader
import numpy as np
import lib.callback as callback
from lib.models.Sauron import Sauron
from lib.models.nnUNet import nnUNet

def train(cfg: dict, exp_name: str):
    # Set output path = output_path + exp_name
    cfg["path"] = os.path.join(cfg["path"], exp_name)
    # Create output folder if it doesn't exist, and find 'run ID'
    if not os.path.isdir(cfg["path"]):
        os.makedirs(cfg["path"])
        run_id = 1
    else:
        run_folders = [int(x) for x in os.listdir(cfg["path"]) if x.isdigit()]
        if len(run_folders) > 0:
            run_id = np.max(run_folders)+1
        else:
            run_id = 1
    cfg["path"] = os.path.join(cfg["path"], str(run_id))
    os.makedirs(cfg["path"])

    Log(os.path.join(cfg["path"], "config.json")).saveConfig(cfg)
    log = Log(os.path.join(cfg["path"], "log.txt"))

    # Train nnUNet
    model = Sauron(**cfg["architecture"])
    dataset = cfg["data"]

    log(f"Starting {exp_name} "
        f"(run={run_id})")

    model.initialize(cfg["device"], cfg["model_state"], log, isSauron=True)
    data = dataset(cfg["split"], cfg["seed_data"])
    t0 = time.time()

    if cfg["epochs"] > 0:
        tr_data = data.get("train")
        val_data = data.get("validation")

        if len(tr_data) > 0:
            # DataLoaders. Note that shuffle=False because I randomize it myself
            # Note that num_workers=0 because I add this in the Queue
            tr_loader = DataLoader(tr_data, batch_size=cfg["batch_size"],
                    shuffle=False, pin_memory=False, num_workers=0)

            optimizer = cfg["optim"](model.parameters(), **cfg["optim_params"])
            if "scheduler" in cfg:
                scheduler = cfg["scheduler"](optimizer, **cfg["scheduler_params"])
            else:
                scheduler = None

            # Folder for saving the trained models
            os.makedirs(f"{cfg['path']}/models")

            model.fit(tr_loader=tr_loader, val_data=val_data,
                    epoch_start=cfg["epoch_start"],
                    epochs=cfg["epochs"], val_interval=cfg["val_interval"],
                    loss=cfg["loss"],
                    val_batch_size=cfg["batch_size"], opt=optimizer,
                    scheduler=scheduler,
                    callbacks=cfg["callbacks"],
                    log=log, history=cfg["history"])

    log(f"Total running time - {np.round((time.time()-t0)/3600, 3)} hours")
    log("End")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Gather all input arguments.
    # --dataset is mandatory
    # Dataset "files" (located in lib/data) contain their optimal data aug.
    # and optimization strategy and UNet architecture to achieve SOTA.
    # If that information is specified, the given argument will be used.
    # For example, --epochs 30 will force the training script to run for 30 epochs.
    cfg = parseArguments()

    # You can force or add custom config here. Example:
    # cfg["train.epochs"] = 999
    # cfg["train.new_config"] = "test"
    cfg["callbacks"] = [
            callback._start_training_generate_matrices_csgd,
            #callback._end_epoch_save_all_FMs,
            #callback._end_epoch_prune,
            callback._after_compute_grads_csgd,
            callback._end_epoch_examine_clusters_csgd,
            callback._end_epoch_save_history,
            callback._end_epoch_track_number_filters,
            callback._end_epoch_save_last_model,
            callback._end_training_prune_csgd,
            ]
    #exp_name = f"Sauron_reimp/interpretability/{cfg['data'].name}/"
    exp_name = f"centripetal/{cfg['data'].name}/"
    #exp_name = f"delete_centri/{cfg['data'].name}/"

    train(cfg, exp_name)
