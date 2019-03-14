import os
from pathlib import Path
import copy
import torch
import glob

def rotating_save_checkpoint(state, prefix, path="./checkpoints", nb=5):
    from pathlib import Path
    
    if not os.path.isdir(path):
        os.makedirs(path)
    filenames = []
    first_empty = None
    for i in range(nb):
        filename = Path(path) / f"{prefix}_{i}.pth"
        if not os.path.isfile(filename) and first_empty is None:
            first_empty = filename
        filenames.append(filename)
    
    if first_empty is not None:
        torch.save(state, first_empty)
    else:
        first = filenames[0]
        os.remove(first)
        for filename in filenames[1:]:
            os.rename(filename, first)
            first = filename
        torch.save(state, filenames[-1])

        
            
def load_best_models(base_model, nb, prefix, path="./checkpoints", extract_model = lambda d: d["model"]):
    models = []
    filenames = []
    
    filters = str(Path(path) / f"{prefix}_*.pth")
    files = glob.glob(filters)
    print(f"Extracting best model from {filters}")

    files.sort(reverse = True)

    for file in files[:nb]:
        print(f"Building model from {file}")
        model = copy.deepcopy(base_model)
        model.load_state_dict(extract_model(torch.load(file)))
        models.append(model)
    
    return models

    