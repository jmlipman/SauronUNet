import argparse, os, importlib, time, inspect
from lib.utils import getDataset, Log, getPCname
from lib.paths import data_path
import numpy as np
from lib.models.nnUNet import nnUNet
import pandas as pd

parser = argparse.ArgumentParser(description="Parser for preprocessing datasets")
parser.add_argument("--data", help="Name of the dataset", required=True)
parser.add_argument("--model_state", help="Model's parameters", required=True)
parser.add_argument("--output", help="Output folder", required=True)
parser.add_argument("--original", help="Original data folder", default="")
parser.add_argument("--in_filters", help="File containing in_filters", default="")
parser.add_argument("--out_filters", help="File containing out_filters", default="")

args = parser.parse_args()
outputFolder = args.output
originalFolder = args.original
modelState = args.model_state
datasetName = args.data
inFilters = args.in_filters
outFilters = args.out_filters

# Verify info given by the user
# Which datasets can be preprocessed
#available_datasets = [x.replace(".py", "")
#        for x in os.listdir("lib/data/process") if x.endswith(".py")]
available_datasets = {}
pythonFiles = [x.replace(".py", "") for x in os.listdir("lib/data") if x.endswith(".py")]
for pyfi in pythonFiles:
    for name, cl in inspect.getmembers(importlib.import_module(f"lib.data.{pyfi}")):
        if inspect.isclass(cl):
            if hasattr(cl, "name"):
                available_datasets[getattr(cl, "name")] = cl

if not datasetName in available_datasets:
    raise ValueError(f"--data `{datasetName}` is invalid. Available datasets:"
            f" {available_datasets}")

# Validate model state
if not os.path.isfile(modelState):
    raise ValueError(f"--model_state `{modelState}` does not exist.")

if os.path.isfile(inFilters) and os.path.isfile(outFilters):
    df_in = pd.read_csv(inFilters, sep="\t")
    df_out = pd.read_csv(outFilters, sep="\t")
    filters = {}
    filters["in"] = {col_name: df_in[col_name].iloc[-1] for col_name in df_in.columns}
    filters["out"] = {col_name: df_out[col_name].iloc[-1] for col_name in df_out.columns}
else:
    filters = {}

# Create output folder in the same location where the images are
if not os.path.isdir(outputFolder):
    os.makedirs(outputFolder)
    if originalFolder != "":
        os.makedirs(outputFolder+"_post") # postprocessed
elif len(os.listdir(outputFolder)):
    raise ValueError(f"--output `{outputFolder}` is not empty")

# Predictions
t0 = time.time()
# Empty log -> doesn't save anything
log = Log("")
# The spit ratio is only for training and validation data, so it's irrelevant
data = getDataset(datasetName)
pc_name = getPCname()
data.data_path = data_path[datasetName][pc_name]
data = data([0.5, 0.5], 42)
test_data = data.get("test")

if len(test_data) > 0:
    model = nnUNet(modalities=data.modalities,
                   n_classes=len(data.classes),
                   fms_init=data.opt["architecture"]["fms_init"],
                   levels=data.opt["architecture"]["levels"],
                   normLayer=data.opt["architecture"]["normLayer"],
                   dim=data.opt["architecture"]["dim"],
                   filters=filters
                   )
    model.initialize(device="cuda", model_state=modelState, log=log)
    model.predict(test_data, batch_size=1,
            path_output=outputFolder, path_original=originalFolder)
    print(f"Total running time: {np.round((time.time()-t0)/60, 3)} minutes")
    if originalFolder != "":
        print("Final arrangements")
        data.finalArrangements(outputFolder+"_post", originalFolder)
else:
    raise Exception("No test data (empty data.get('test'))")
