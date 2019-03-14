import os
from pathlib import Path
import torch.nn as nn

from utils.rotating_checkpoint import load_best_models

        
class SnapshotEnsemble(nn.Module):
    def __init__(self, models):
        assert len(models) > 0
        super(SnapshotEnsemble, self).__init__()
        self.models = models
        
    def forward(self, x):
        acc = self.models[0](x)   
        for model in self.models[1:]:
            acc += model(x)
        acc /= len(self.models)
        return acc

    @staticmethod
    def build_from_checkpoints(
        base_model, cycles, prefix, path="./checkpoints",
    ):
        models = []
        for i in cycles:
            model = load_best_models(base_model, 1, prefix, path=Path(path) / prefix / f"cycle_{i}")[0]
            models.append(model)
            
        return SnapshotEnsemble(models)
