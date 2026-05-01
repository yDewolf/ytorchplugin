# This is really cool, I love when I can do things like this (Rust made my mind)
from typing import TypeVar, Generic
import logging

INSTRUCTION_LOGGER = logging.Logger("Instruction")

VARIABLE_ACCESS: str = "$"
PATH_SEPARATOR: str = ":"

T = TypeVar("T")

class InstructionVariable(Generic[T]):
    var_path: str

    target_key: str
    raw_value: T | None = None # TODO: Puxar isso aqui com base na key

    def __init__(self, var_path: str, raw_value: T | None, target_key: str = "") -> None:
        self.var_path = var_path
        self.target_key = target_key
        self.raw_value = raw_value

    @staticmethod
    def make_variable_path(path: list[str]):
        return f"${PATH_SEPARATOR.join(path)}"


    # Path must include variable name
    @staticmethod
    def from_data_variable(path: list[str], value: T):
        var_path: str = PATH_SEPARATOR.join(path)
        target_key: str = ""
        raw_value: T | None = value

        if type(value) is str:
            if value.startswith(VARIABLE_ACCESS):
                target_key = value
                raw_value = None

        return InstructionVariable[T](
            var_path,
            raw_value,
            target_key=target_key
        )
    

    def __str__(self) -> str:
        return f"Var({self.var_path}={self.raw_value if not self.raw_value is None else self.target_key})"


class VariableController:
    variables: dict[str, InstructionVariable]

    def __init__(self) -> None:
        self.variables = {}

    def add_object_variables(self, obj: object, deep: bool = False):
        for attribute_name in obj.__dict__:
            attribute = obj.__getattribute__(attribute_name)

            if type(attribute) is InstructionVariable:
                self.add_variable(attribute, deep)


    def add_variable(self, variable: InstructionVariable, deep: bool = False):
        self.set_variable(variable)

        if variable.raw_value is None:
            return
        
        if deep and hasattr(variable.raw_value, "__dict__"):
            for attribute_name in variable.raw_value.__dict__:
                attribute = variable.raw_value.__getattribute__(attribute_name)

                if type(attribute) is InstructionVariable:
                    self.add_variable(attribute, deep)
    
    def set_variable(self, variable: InstructionVariable):
        self.variables[variable.var_path] = variable


    def get_var_from_path(self, var_path: str) -> InstructionVariable | None:
        var_path = VariableController._get_var_path_without_prefix(var_path)
        variable: InstructionVariable | None = self.variables.get(var_path)
        
        return variable

    def get_variable(self, var_path: str):
        variable = self.get_var_from_path(var_path)
        if variable is None:
            INSTRUCTION_LOGGER.error(f"ERROR: Couldn't find {var_path} in Controller {self}")
            return

        value = variable.raw_value
        if variable.target_key != "":
            value = self.get_variable(variable.target_key)
        
        return value

    def get_non_none_value(self, variable: InstructionVariable):
        value = self.get_variable(variable.var_path)
        if value is None:
            INSTRUCTION_LOGGER.error(f"ERROR: Variable shouldn't have None value - Var: {variable}")
            return Exception("Non none value is None")

        return value

    @staticmethod
    def _get_var_path_without_prefix(var_path: str):
        return var_path.removeprefix(VARIABLE_ACCESS)