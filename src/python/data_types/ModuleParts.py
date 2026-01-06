import torch.nn as nn
from torch.nn import Module

class ModulePart:
    module_type: type[Module]
    torch_module: Module

    is_setup: bool = False

    def __init__(self, module_type: type[Module]) -> None:
        self.module_type = module_type

    def setup_module(self, params: list):
        self.torch_module = self.module_type(params)
        self.is_setup = True


    @staticmethod
    def from_type_str(type_str: str):
        match type_str.lower():
            case "linear":
                return ModulePart(nn.Linear)
            
            case "conv1d":
                return ModulePart(nn.Conv1d)
            
            case "conv2d":
                return ModulePart(nn.Conv2d)