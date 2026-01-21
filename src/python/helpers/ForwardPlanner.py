
from enum import Enum
import logging

import torch
from data_types.ModuleParts import ModulePart
from data_types.InstructionVariables import InstructionVariable, VariableController
from helpers.ShapeCalculator import ShapeCalculator

FORWARD_LOGGER = logging.Logger("ForwardLogger")

class OperationKeys(Enum):
    UNKNOWN = -1
    MODULE_FORWARD = 0
    RESHAPE = 1
    PERMUTE = 2

    @staticmethod
    def from_str(value_str: str | None):
        if value_str is None:
            return OperationKeys.UNKNOWN
        
        match value_str.lower():
            case "module_forward": return OperationKeys.MODULE_FORWARD
            case "reshape": return OperationKeys.RESHAPE
            case "permute": return OperationKeys.PERMUTE
            case _: return OperationKeys.UNKNOWN


class ForwardOperation:
    valid: bool = False

    operation_type: str | None
    operation_key: OperationKeys
    raw_params: dict
    params: dict[str, InstructionVariable]

    def __init__(self, operation_type: str | None, params: dict) -> None:
        self.operation_type = operation_type
        self.operation_key = OperationKeys.from_str(self.operation_type)
        self.raw_params = params
    
        if operation_type == None:
            self.valid = False
            return

        self.valid = True

    @staticmethod
    def from_dict(data_dict: dict):
        operation_type = data_dict.get("operation")
        params = data_dict.get("params", {})

        return ForwardOperation(
            operation_type,
            params
        )
    
    def parse_parameters(self, var_controller: VariableController, path: list[str] = ["operations"]):
        parsed_parameters = {}
        for key in self.raw_params:
            var = InstructionVariable.from_data_variable(path + [key], self.raw_params[key])
            parsed_parameters[key] = var
        
        return parsed_parameters
    


class ForwardPlanner:
    modules: list[ModulePart]
    forward_operations: list[ForwardOperation]
    global_var_controller: VariableController

    internal_var_controller: VariableController

    def __init__(self, modules: list[ModulePart], forward_operations: list[ForwardOperation], global_var_controller: VariableController) -> None:
        self.modules = modules
        self.forward_operations = forward_operations
        self.global_var_controller = global_var_controller

        self.internal_var_controller = VariableController()
        self._setup_operations()
        self._setup_modules()


    def _setup_operations(self):
        for idx, operation in enumerate(self.forward_operations):
            parsed = operation.parse_parameters(self.global_var_controller, path=["operations", str(idx)])
            operation.params = parsed

            for key, var in operation.params.items():
                self.internal_var_controller.add_variable(var)
    
    def _setup_modules(self):
        for module in self.modules:
            var = InstructionVariable.from_data_variable(
                ["modules", module.key],
                module
            )

            self.internal_var_controller.add_variable(var)


    def pre_forward_pass(self, inputs: dict[str, torch.Tensor]):
        # Setup input variables
        for input_key, value in inputs.items():
            self.internal_var_controller.set_variable(
                InstructionVariable.from_data_variable(
                    ["inputs", input_key], value
                )
            )



    def forward_pass(self):
        for idx, operation in enumerate(self.forward_operations):
            match operation.operation_key:
                case OperationKeys.UNKNOWN: return Exception("Unknown Operation Type")
                case OperationKeys.MODULE_FORWARD:
                    module_pointer: InstructionVariable | None = operation.params.get("module", None)
                    if module_pointer is None:
                        FORWARD_LOGGER.error(f"ERROR: Couldn't find module_var for operation {idx} - Params: {operation.params}")
                        break

                    module_part = self.internal_var_controller.get_non_none_value(module_pointer)
                    if not type(module_part) is ModulePart:
                        FORWARD_LOGGER.error(f"ERROR: Wrong type for module part var on operation {idx} - Params: {operation.params}")
                        break
                    
                    # Check if the operation targets a Input
                    # otherwise we just get the model's input pointer
                    internal_input_var: InstructionVariable | None = operation.params.get("input", None)
                    if internal_input_var is None:
                        # This InstructionVariable points to an internal variable in the forward runtime through the target key
                        # its raw_value should always be None
                        global_input_pointer = self.global_var_controller.get_var_from_path(f"$modules:{module_part.key}:input")
                        if global_input_pointer is None:
                            FORWARD_LOGGER.error(f"ERROR: Couldn't find any input variable for module {module_part.key} on operation {idx} - Params: {operation.params}")
                            break
                        
                        internal_input_var = InstructionVariable(global_input_pointer.target_key, None)

                    input_value = self.internal_var_controller.get_non_none_value(internal_input_var)
                    if not type(input_value) is torch.Tensor:
                        FORWARD_LOGGER.error(f"ERROR: Module's input is not a Tensor on operation {idx} - Module {module_part.key}")
                        break
                    
                    if not ShapeCalculator.check_input_shape(module_part, input_value.shape):
                        FORWARD_LOGGER.error(f"ERROR: Couldn't pass Tensor to module since Shape Error on operation {idx} - Module {module_part.key}")
                        break

                    model_output = module_part.torch_module(input_value)
                    # Register module output so other modules can use this value
                    self.internal_var_controller.set_variable(
                        InstructionVariable.from_data_variable(["modules", module_part.key, "out"], model_output)
                    )

                case OperationKeys.RESHAPE: 
                    pass
                
                case OperationKeys.PERMUTE: 
                    pass