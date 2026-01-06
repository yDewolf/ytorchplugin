FORMAT_VERSION = 0
VARIABLE_ACCESS: str = "$"
PATH_SEPARATOR: str = ":"

class InstructionVariable:
    var_path: str

    target_key: str
    raw_value: object | None = None # TODO: Puxar isso aqui com base na key

    def __init__(self, var_path: str, raw_value: object | None, target_key: str = "") -> None:
        self.var_path = var_path
        self.target_key = target_key
        self.raw_value = raw_value

    @staticmethod
    def make_variable_name(path: list[str]):
        return f"${PATH_SEPARATOR.join(path)}"


    # Path must include variable name
    @staticmethod
    def from_data_variable(path: list[str], value: object):
        var_path: str = PATH_SEPARATOR.join(path)
        target_key: str = ""
        raw_value: object | None = value

        if type(value) is str:
            if value.startswith(VARIABLE_ACCESS):
                target_key = value
                raw_value = None

        return InstructionVariable(
            var_path,
            raw_value,
            target_key=target_key
        )

# TODO: Fazer module function funcionar como uma variável
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


class ModuleParameters:
    raw_parameters: dict

    in_features: InstructionVariable
    output_size: InstructionVariable

    def __init__(self, param_data: dict, in_features: InstructionVariable, output_size: InstructionVariable) -> None:
        self.in_features = in_features
        self.output_size = output_size


    @staticmethod
    def from_dict(data_dict: dict, module_path: list[str] = []):
        path = module_path + ["params"]
        return ModuleParameters(
            data_dict,
            InstructionVariable.from_data_variable(path + ["in_features"], data_dict["in_features"]),
            InstructionVariable.from_data_variable(path + ["out_features"], data_dict["out_features"])
        )

class ModuleInstructions:
    module_key: str
    module_type: str

    init_parameters: ModuleParameters
    input_key: InstructionVariable
    
    before: ModuleFunction | None
    activation: str | None = None

    def __init__(self, module_key: str, type: str, init_parameters: ModuleParameters, input_key: InstructionVariable, before: ModuleFunction | None, activation: str) -> None:
        self.module_key = module_key
        self.module_type = type
        self.init_parameters = init_parameters
        self.input_key = input_key
        self.before = before
        self.activation = activation

    @staticmethod
    def from_dict(key: str, data_dict: dict, root_path: list[str] = []):
        before = data_dict.get("before", {})
        func: ModuleFunction | None = ModuleFunction.from_dict(before) if before != {} else None
        
        path = root_path + [key]
        return ModuleInstructions( 
            key,
            data_dict["type"],
            ModuleParameters.from_dict(data_dict["params"], module_path=path),
            InstructionVariable.from_data_variable(path + ["input"], data_dict["input"]),
            func,
            data_dict["activation"]
        )


class InstructionHolder:
    version: int = 0
    raw_data: dict

    modules: dict[str, ModuleInstructions]
    functions: list[ModuleFunction]

    def __init__(self, raw_data: dict, modules: dict[str, ModuleInstructions], functions: list[ModuleFunction], version = FORMAT_VERSION) -> None:
        self.version = version
        self.raw_data = raw_data
        
        self.modules = modules
        self.functions = functions
    
    @staticmethod
    def from_dict(data_dict: dict):
        version = data_dict.get("version", -1)
        modules: dict[str, ModuleInstructions] = {}

        modules_data: dict = data_dict.get("modules", {})
        for module_key in modules_data:
            data: dict = modules_data[module_key]

            instructions = ModuleInstructions.from_dict(module_key, data, root_path=["modules"])
            modules[module_key] = instructions
        
        return InstructionHolder(data_dict, modules, [], version=version)