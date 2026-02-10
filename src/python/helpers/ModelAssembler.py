import torch
import torch.nn as nn

from data_types.ModuleInstructions import InstructionHolder, ModuleInstructions
from data_types.ModuleParts import ModulePart
from model.Model import Model
from helpers.ForwardPlanner import ForwardOperation, ForwardPlanner
from data_types.InstructionVariables import VariableController

class ModelAssembler:
    
    @staticmethod
    def assemble_model(instructions: InstructionHolder) -> Model:
        torch_device = torch.device(instructions.device)
        global_var_controller = instructions.variable_controller

        module_parts: list[ModulePart] = []
        for module_instruction in instructions.get_modules():
            module_part = ModulePart.from_type_str(module_instruction.get_module_type()) # type: ignore
            if module_part is None:
                continue
            
            module_part.key = module_instruction.module_key
            parsed_parameters = module_instruction.get_init_parameters(global_var_controller)
            parsed_parameters["device"] = torch_device

            module_part.setup_module(parsed_parameters)
            module_parts.append(module_part)

        forward_planner = ModelAssembler.assemble_forward_planner(
            module_parts,
            instructions.forward_operations,
            global_var_controller
        )
        return Model(
            module_parts,
            forward_planner,
            torch_device
        )
    
    @staticmethod
    def assemble_forward_planner(module_parts: list[ModulePart], forward_operations: list[ForwardOperation], global_var_controller: VariableController) -> ForwardPlanner:
        return ForwardPlanner(
            module_parts,
            forward_operations,
            global_var_controller
        )