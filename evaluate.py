import argparse, os, importlib, inspect, json, time
from lib.metric import Metric
import nibabel as nib
import numpy as np

# Input can be the folder of the train or test set
# It is automatically inferred which one is it.
t0 = time.time()
parser = argparse.ArgumentParser(description="Parser for evaluating data")
parser.add_argument("--data", help="Name of the dataset", required=True)
parser.add_argument("--pred", help="Input folder", required=True)
parser.add_argument("--gt", help="Folder with the original files", required=True)
parser.add_argument("--output", help="Output file", required=True)

args = parser.parse_args()
data = args.data
predFolder = args.pred
gtFolder = args.gt
outputFile = args.output

# Which datasets can be preprocessed
#available_datasets = [x.replace(".py", "")
#        for x in os.listdir("lib/data/process/") if x.endswith(".py")]
available_datasets = {}
pythonFiles = [x.replace(".py", "") for x in os.listdir("lib/data") if x.endswith(".py")]
for pyfi in pythonFiles:
    for name, cl in inspect.getmembers(importlib.import_module(f"lib.data.{pyfi}")):
        if inspect.isclass(cl):
            if hasattr(cl, "name"):
                available_datasets[getattr(cl, "name")] = cl

if not data in available_datasets:
    raise ValueError(f"--data `{data}` is invalid. Available datasets:"
            f" {available_datasets}")


# If these folders don't exist, raise Exception
for path, option in zip([predFolder, gtFolder], ["pred", "gt"]):
    if not os.path.isdir(path):
        raise ValueError(f"--{option} `{path}` must be an existing folder.")

if os.path.isfile(outputFile):
    raise ValueError(f"--output file `{outputFile}` already exists.")

C = available_datasets[data]

# Lists containing the files that will be compared
predFiles = C.findPredictionFiles(predFolder)
GTFiles = C.findGroundTruthFiles(gtFolder, predFiles)

if len(GTFiles) != len(predFiles):
    raise ValueError("For some reason, the number of predictions is different "
            "to the number of ground-truth files")

metrics = ["dice", "HD", "TFPN", "surface_dice"]
metrics = ["dice", "HD"]
print(f"Computing the following metrics: {metrics}")
Measure = Metric(metrics, onehot=C.onehot,
        classes=C.classes,
        classes_mean=C.measure_classes_mean,
        multiprocess=True)

results = {}
for i, (pred_path, gt_path) in enumerate(zip(predFiles, GTFiles)):
    print(f"Loading: {i+1}/{len(GTFiles)}")
    pred = nib.load(pred_path)
    voxelspacing = pred.header.get_zooms()
    pred = pred.get_fdata()
    gt = nib.load(gt_path).get_fdata()

    # Convert to one-hot vectors
    pred = np.stack([pred==i for i in sorted(C.classes)])
    gt = np.stack([gt==i for i in sorted(C.classes)])
    #from IPython import embed; embed()

    #from IPython import embed; embed()
    #raise Exception("para")
    sub_id = pred_path.split("/")[-1].replace(".nii.gz", "")
    results[sub_id] = Measure.all(pred, gt, {"voxelspacing": voxelspacing})

for i, k in enumerate(results):
    print(f"Computing: {i+1}/{len(GTFiles)}")
    results[k] = results[k].get()
Measure.close()

# Compute average
metrics = list(results[k].keys())
average = np.zeros((len(metrics), len(C.classes)))
for m in range(len(metrics)):
    for c in range(len(C.classes)):
        average[m, c] = np.mean([results[s][metrics[m]][c] for s in results])

results["average"] = {}
for m, metric in enumerate(metrics):
    results["average"][metric] = list(average[m])

with open(outputFile, "w") as f:
    f.write(json.dumps(results))

print(f"Total time: {np.round((time.time()-t0)/60, 3)} mins.")
