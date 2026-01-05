
class ModuleFunction:
    type: str
    args: list

    def __init__(self, type: str, args: list) -> None:
        self.type = type
        self.args = args

    @staticmethod
    def from_dict(data_dict: dict):
        return ModuleFunction(
            data_dict["function"],
            data_dict["arguments"],
        )
    

class ModuleInstructions:
    module_key: str
    module_type: str

    init_parameters: list
    input_key: str
    
    before: ModuleFunction | None
    activation: str | None = None

    def __init__(self, module_key: str, type: str, init_parameters: list, input_key: str, before: ModuleFunction | None, activation: str) -> None:
        self.module_key = module_key
        self.module_type = type
        self.init_parameters = init_parameters
        self.input_key = input_key
        self.before = before
        self.activation = activation

    @staticmethod
    def from_dict(key: str, data_dict: dict):
        before = data_dict.get("before", {})
        func: ModuleFunction | None = ModuleFunction.from_dict(before) if before != {} else None
        return ModuleInstructions( 
            key,
            data_dict["type"],
            data_dict["params"],
            data_dict["input"],
            func,
            data_dict["activation"]
        )


class InstructionAssembler:
    modules: list[ModuleInstructions]
    functions: list[ModuleFunction]

    def __init__(self) -> None:
        pass