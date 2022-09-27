import argparse, os, importlib, inspect

parser = argparse.ArgumentParser(description="Parser for preprocessing datasets")
parser.add_argument("--data", help="Name of the dataset", required=True)
parser.add_argument("--input", help="Input folder", required=True)
parser.add_argument("--original", help="Folder with the original files", default="")
parser.add_argument("--output", help="Output folder", required=True)

args = parser.parse_args()
data = args.data
inputFolder = args.input
originalFolder = args.original
outputFolder = args.output

### 1) VALIDATE AND SANITIZE INPUT
# Confirm that the chosen dataset is available
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

# Input folder must exist and not be empty
if not os.path.isdir(inputFolder):
    raise ValueError(f"--input `{inputFolder}` must be an existing folder.")
elif len(os.listdir(inputFolder)) == 0:
    raise ValueError(f"--input `{inputFolder}` is empty.")

# If output folder exists, it must be empty
if os.path.isdir(outputFolder) and len(os.listdir(outputFolder)) > 0:
    raise ValueError(f"--ouptut `{outputFolder}` must not exist, or it must be empty.")

### 2) FIND VALIDATION AND PROCESSING FUNCTIONS
C = available_datasets[data]

### 3) VERIFY AND PROCESS
print(f"Verifying dataset `{data}`")
if C.pre_verify(inputFolder):

    # Create output folder in the same location where the images are
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)
    elif len(os.listdir(outputFolder)):
        raise ValueError(f"--output `{outputFolder}` is not empty")

    C.pre_process(inputFolder, outputFolder, args.original)
else:
    raise Exception("There was a problem with the data. Read the warnings.")
