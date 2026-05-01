import torch
from data_types.ModuleParts import ModulePart
from helpers.ForwardPlanner import ForwardPlanner

class Model:
    device: torch.device

    module_parts: list[ModulePart]
    forward_planner: ForwardPlanner

    def __init__(self, module_parts: list[ModulePart], forward_planner: ForwardPlanner, device: torch.device) -> None:
        self.module_parts = module_parts
        self.forward_planner = forward_planner
        self.device = device

    
    def forward(self, inputs: dict[str, torch.Tensor], safe_mode: bool = True) -> bool:
        successful, missing_inputs, invalid_inputs = self.forward_planner.pre_forward_pass(inputs)
        if not successful:
            return False
        
        self.forward_planner.forward_pass(safe_mode)
        self.forward_planner.post_forward_pass()

        return True

    def backward(self):
        pass

    def compute_loss(self):
        pass