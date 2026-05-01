import logging
from data_types.InstructionVariables import VARIABLE_ACCESS, InstructionVariable, VariableController
from helpers.ForwardPlanner import ForwardOperation

FORMAT_VERSION = 0
INSTRUCTION_LOGGER = logging.Logger("InstructionLogger")

class ModuleFunction:
    type: str
    args: dict

    def __init__(self, type: str, args: dict) -> None:
        self.type = type
        self.args = args

    @staticmethod
    def from_dict(data_dict: dict):
        return ModuleFunction(
            data_dict["function"],
            data_dict["arguments"],
        )

    def __str__(self) -> str:
        args_str: list[str] = []
        for key in self.args:
            args_str.append(f"{key}={self.args[key]}")

        return f"{self.type}({", ".join(args_str)})"

class ModuleParameters:
    raw_parameters: dict

    in_features: InstructionVariable
    output_size: InstructionVariable

    def __init__(self, param_data: dict, in_features: InstructionVariable, output_size: InstructionVariable) -> None:
        self.raw_parameters = param_data
        self.in_features = in_features
        self.output_size = output_size


    @staticmethod
    def from_dict(data_dict: dict, module_path: list[str] = []):
        path = module_path + ["data"]
        return ModuleParameters(
            data_dict,
            InstructionVariable.from_data_variable(path + ["in_features"], data_dict["in_features"]),
            InstructionVariable.from_data_variable(path + ["out_features"], data_dict["out_features"])
        )

class ModuleInstructions:
    module_key: str
    module_type: InstructionVariable

    init_parameters: InstructionVariable[ModuleParameters]
    input_key: InstructionVariable
    
    # before: InstructionVariable[ModuleFunction] | None
    activation: InstructionVariable

    def __init__(self, module_key: str, type: InstructionVariable, init_parameters: InstructionVariable[ModuleParameters], input_key: InstructionVariable, activation: InstructionVariable) -> None:
        self.module_key = module_key
        self.module_type = type
        self.init_parameters = init_parameters
        self.input_key = input_key
        # self.before = before
        self.activation = activation

    @staticmethod
    def from_dict(key: str, data_dict: dict, root_path: list[str] = []):
        # before = data_dict.get("before", {})
        path = root_path + [key]

        # func: InstructionVariable[ModuleFunction] | None = InstructionVariable[ModuleFunction].from_data_variable(
        #     path + ["before"],    
        #     ModuleFunction.from_dict(before) 
        # ) if before != {} else None
        
        return ModuleInstructions( 
            key,
            InstructionVariable.from_data_variable(path + ["type"], data_dict["type"]),
            InstructionVariable[ModuleParameters].from_data_variable(
                path, 
                ModuleParameters.from_dict(data_dict["data"], module_path=path)
            ),
            InstructionVariable.from_data_variable(path + ["input"], data_dict["input"]),
            # func,
            InstructionVariable.from_data_variable(path + ["activation"], data_dict.get("activation", None)),
        )

    def get_module_type(self) -> str:
        if self.module_type.raw_value is None:
            return ""

        return self.module_type.raw_value

    def get_init_parameters(self, controller: VariableController) -> dict:
        if self.init_parameters.raw_value is None:
            return {}

        parameters: dict = {}
        for param_key, value in self.init_parameters.raw_value.raw_parameters.items():
            real_value = value
            if type(value) is str and value.startswith(VARIABLE_ACCESS):
                real_value = controller.get_variable(real_value)
            
            parameters[param_key] = real_value
        
        return parameters



    def __str__(self) -> str:
        return f"Module({self.module_key}, {self.module_type})"


class InstructionHolder:
    raw_data: dict
    version: int = 0
    device: str = "cpu"

    forward_operations: list[ForwardOperation]
    modules: dict[str, InstructionVariable[ModuleInstructions]]
    functions: list[ModuleFunction]
    variable_controller: VariableController

    def __init__(self, raw_data: dict, modules: dict[str, InstructionVariable[ModuleInstructions]], forward_operations: list[ForwardOperation], functions: list[ModuleFunction], version = FORMAT_VERSION, device = "cpu", controller: VariableController = VariableController()) -> None:
        self.version = version
        self.device = device
        self.raw_data = raw_data
        
        self.modules = modules
        self.forward_operations = forward_operations
        self.functions = functions
        self.variable_controller = controller
    
    @staticmethod
    def from_dict(data_dict: dict):
        version = data_dict.get("version", -1)
        device = data_dict.get("device", "cpu")
        modules: dict[str, InstructionVariable[ModuleInstructions]] = {}

        modules_data: dict = data_dict.get("modules", {})
        modules_path: list[str] = ["modules"]
        var_controller: VariableController = VariableController()

        for module_key in modules_data:
            data: dict = modules_data[module_key]

            instructions = ModuleInstructions.from_dict(module_key, data, root_path=modules_path)
            var = InstructionVariable.from_data_variable(
                modules_path + [module_key],
                instructions
            )

            var_controller.add_variable(var)
            var_controller.add_object_variables(instructions, True)
            modules[module_key] = var
        
        forward_data = data_dict.get("forward", [])
        # forward_path: list[str] = ["forward"]
        forward_operations: list[ForwardOperation] = []
        for operation_data in forward_data:
            operation = ForwardOperation.from_dict(operation_data)
            
            # var = InstructionVariable.from_data_variable(
            #       forward_path
            # )
            # var_controller.add_object_variables(operation.raw_params)

            forward_operations.append(operation)
        
        return InstructionHolder(data_dict, modules, forward_operations, [], version=version, device=device, controller=var_controller)
    

    def get_modules(self) -> list[ModuleInstructions]:
        modules: list[ModuleInstructions] = []
        for variable in self.modules.values():
            if variable.raw_value is None:
                INSTRUCTION_LOGGER.error("ERROR: Module variable has None value instead of ModuleInstruction")
                continue

            modules.append(variable.raw_value)
        
        return modules



    def get_variable(self, path: str):
        return self.variable_controller.get_variable(path)