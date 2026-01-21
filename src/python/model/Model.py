import torch
from data_types.ModuleParts import ModulePart
from helpers.ForwardPlanner import ForwardPlanner

class Model:
    module_parts: list[ModulePart]
    forward_planner: ForwardPlanner

    def __init__(self, module_parts: list[ModulePart], forward_planner: ForwardPlanner) -> None:
        self.module_parts = module_parts
        self.forward_planner = forward_planner

    
    def forward(self, inputs: dict[str, torch.Tensor]) -> bool:
        successful, missing_inputs, invalid_inputs = self.forward_planner.pre_forward_pass(inputs)
        if not successful:
            return False
        
        self.forward_planner.forward_pass()
        self.forward_planner.post_forward_pass()

        return True

    def backward(self):
        pass

    def compute_loss(self):
        pass