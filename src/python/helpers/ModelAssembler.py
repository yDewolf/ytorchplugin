import torch
import torch.nn as nn

from data_types.ModuleInstructions import InstructionHolder
from data_types.ModuleParts import ModulePart

class Model:
    module_parts: list[ModulePart]

    def __init__(self, module_parts: list[ModulePart]) -> None:
        self.module_parts = module_parts

    
    def forward(self, input: torch.Tensor):
        pass

    def backward(self):
        pass

    def compute_loss(self):
        pass


class ModelAssembler:
    
    @staticmethod
    def assemble_model(instructions: InstructionHolder) -> Model:
        torch_device = torch.device(instructions.device)
        var_controller = instructions.variable_controller

        module_parts: list[ModulePart] = []
        for module_instruction in instructions.get_modules():
            module_part = ModulePart.from_type_str(module_instruction.get_module_type()) # type: ignore
            if module_part is None:
                continue
            

            parsed_parameters = module_instruction.get_init_parameters(var_controller)
            parsed_parameters["device"] = torch_device

            module_part.setup_module(parsed_parameters)
            module_parts.append(module_part)
        
        return Model(
            module_parts
        )