import torch
from data_types.ModuleParts import ModulePart
from helpers.ForwardPlanner import ForwardPlanner

class Model:
    module_parts: list[ModulePart]
    forward_planner: ForwardPlanner

    def __init__(self, module_parts: list[ModulePart], forward_planner: ForwardPlanner) -> None:
        self.module_parts = module_parts
        self.forward_planner = forward_planner

    
    def forward(self, input: torch.Tensor):
        pass

    def backward(self):
        pass

    def compute_loss(self):
        pass