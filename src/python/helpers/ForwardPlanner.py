
from enum import Enum
import logging

import torch
from data_types.ModuleParts import DimShape, ModulePart
from data_types.InstructionVariables import PATH_SEPARATOR, InstructionVariable, VariableController
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

    def __str__(self) -> str:
        return f"FwOperation(key={self.operation_key}, params={self.raw_params})"

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


    def pre_forward_pass(self, inputs: dict[str, torch.Tensor]) -> tuple[bool, list[str], list[str]]:
        input_is_valid, missing_inputs, invalid_inputs = self.validate_inputs(inputs)
        if not input_is_valid:
            return False, missing_inputs, invalid_inputs
        
        # Setup input variables
        for input_key, value in inputs.items():
            self.internal_var_controller.set_variable(
                InstructionVariable.from_data_variable(
                    ["inputs", input_key], value
                )
            )
        
        return True, [], []

    def forward_pass(self, safe: bool = True):
        for idx, operation in enumerate(self.forward_operations):
            match operation.operation_key:
                case OperationKeys.UNKNOWN: return Exception("Unknown Operation Type")
                case OperationKeys.MODULE_FORWARD:
                    successful = self._module_forward(idx, operation, safe)
                    break

                case _:
                    successful = self._run_operation_function(idx, operation, safe)
            
    def post_forward_pass(self):
        # probably some cleanup
        pass


    def get_required_inputs(self) -> dict[str, ModulePart]:
        # TODO: Get all input keys model needs + its shapes 
        required_inputs: dict[str, ModulePart] = {}
        for module in self.modules:
            input_pointer: InstructionVariable | None = self.global_var_controller.get_var_from_path(InstructionVariable.make_variable_path([
                "modules", module.key, "input"
            ]))
            if input_pointer is None:
                FORWARD_LOGGER.error(f"ERROR: Module doesn't have a input pointer - Module {module.key}")
                break
            
            clean_input_path = VariableController._get_var_path_without_prefix(input_pointer.target_key)
            receives_external_input = clean_input_path.startswith("inputs")
            if not receives_external_input:
                continue
            
            required_inputs[PATH_SEPARATOR.join(clean_input_path.split(PATH_SEPARATOR)[1:])] = module
            
        return required_inputs

    def validate_inputs(self, inputs: dict[str, torch.Tensor]) -> tuple[bool, list[str], list[str]]:
        required_inputs: dict[str, ModulePart] = self.get_required_inputs()
        
        missing_inputs: list[str] = []
        invalid_inputs: list[str] = []
        for input_key, module in required_inputs.items():
            input_value = inputs.get(input_key, None)
            if input_value is None: 
                missing_inputs.append(input_key)
                continue
            
            if not ShapeCalculator.check_module_input_shape(module, input_value.shape):
                invalid_inputs.append(input_key)
                continue
        
        return len(missing_inputs) == 0 and len(invalid_inputs) == 0, missing_inputs, invalid_inputs


    def _module_forward(self,  idx: int, operation: ForwardOperation, safe: bool = True) -> bool:
        module_pointer: InstructionVariable | None = operation.params.get("module", None)
        if module_pointer is None:
            FORWARD_LOGGER.error(f"ERROR: Couldn't find module_var for operation {idx} - Params: {operation.params}")
            return False

        module_part = self.internal_var_controller.get_non_none_value(module_pointer)
        if not type(module_part) is ModulePart:
            FORWARD_LOGGER.error(f"ERROR: Wrong type for module part var on operation {idx} - Params: {operation.params}")
            return False
        
        # Check if the operation targets a Input
        # otherwise we just get the model's input pointer
        internal_input_var: InstructionVariable | None = operation.params.get("input", None)
        if internal_input_var is None:
            # This InstructionVariable points to an internal variable in the forward runtime through the target key
            # its raw_value should always be None
            global_input_pointer = self.global_var_controller.get_var_from_path(f"$modules:{module_part.key}:input")
            if global_input_pointer is None:
                FORWARD_LOGGER.error(f"ERROR: Couldn't find any input variable for module {module_part.key} on operation {idx} - Params: {operation.params}")
                return False
            
            internal_input_var = InstructionVariable(global_input_pointer.target_key, None)

        input_value = self.internal_var_controller.get_non_none_value(internal_input_var)
        if safe:
            if not type(input_value) is torch.Tensor:
                FORWARD_LOGGER.error(f"ERROR: Module's input is not a Tensor on operation {idx} - Module {module_part.key}")
                return False
            
            if not ShapeCalculator.check_module_input_shape(module_part, input_value.shape):
                FORWARD_LOGGER.error(f"ERROR: Couldn't pass Tensor to module since Shape Error on operation {idx} - Module {module_part.key}")
                return False

        model_output = module_part.torch_module(input_value)
        # Register module output so other modules can use this value
        self.internal_var_controller.set_variable(
            InstructionVariable.from_data_variable(["modules", module_part.key, "out"], model_output)
        )
        
        return True
    
    def _run_operation_function(self, idx: int, operation: ForwardOperation, safe: bool) -> bool:
        successful = False
        match operation.operation_type:
            case OperationKeys.RESHAPE:
                FORWARD_LOGGER.debug(f"Running reshape operation {operation}")
        
            case OperationKeys.PERMUTE:
                FORWARD_LOGGER.debug(f"Running permute operation {operation}")

        return successful
