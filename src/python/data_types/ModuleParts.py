import logging
from torch.nn import Module
import torch
import torch.nn as nn
import random
from enum import Enum

PART_LOGGER = logging.Logger("PartLogger")

DIM_ANY_VALUE = -1
INT32_LIMIT = 2 ** 31

class DimShape:
    dim_count: int = 1
    shape: tuple = (1, )
    size_range: tuple[int, int] = (1, INT32_LIMIT)

    def __init__(self, shape: tuple, dim_count: int, size_range: tuple[int, int] = (1, INT32_LIMIT)) -> None:
        self.dim_count = dim_count
        self.shape = shape
        self.size_range = size_range

    @staticmethod
    def from_module(module: Module):
        with torch.no_grad():
            parameter = next(module.parameters())
            base_shape = parameter.shape
            shape: list[int] = [value for value in base_shape]

            if issubclass(type(module), torch.nn.modules.conv._ConvNd):
                for idx in range(2, len(shape)):
                    shape[idx] = DIM_ANY_VALUE
                
                
                return DimShape(
                    tuple(shape[1:]),
                    len(shape[1:]),
                    (max(base_shape[2:]), INT32_LIMIT)
                )
            
            if type(module) is torch.nn.modules.linear.Linear:
                shape[0] = DIM_ANY_VALUE

            return DimShape(tuple(shape), len(shape))
    


    def sample(self, use_random: bool = False, random_range: tuple[int, int] = (1, 16)):
        random_range = (max(self.size_range[0], random_range[0]), min(self.size_range[1], random_range[1]))

        shape: list[int] = []
        for dim_idx in range(self.dim_count):
            if self.shape[dim_idx] == DIM_ANY_VALUE:
                shape.append(self.size_range[0])
                continue

            shape.append(self.shape[dim_idx] if not use_random else random.randint(random_range[0], random_range[1]))

        return tuple(shape)
    
    def is_compatible(self, shape: tuple) -> tuple[bool, list[int]]:
        compatbile: bool = True
        wrong_dims: list[int] = []

        for idx in range(self.dim_count):
            if self.shape[idx] != DIM_ANY_VALUE and self.shape[idx] != shape[idx]:
                compatbile = False
                wrong_dims.append(idx)
                continue

            if shape[idx] > self.size_range[1]:
                compatbile = False
                wrong_dims.append(idx)
                continue
            
            if shape[idx] < self.size_range[0]:
                compatbile = False
                wrong_dims.append(idx)
                continue
        
        compatbile = compatbile and True
        return (compatbile, wrong_dims)



class ModulePart:
    module_type: type[Module]
    key: str

    torch_module: Module

    is_setup: bool = False
    setup_arguments: dict

    _input_shape: DimShape

    def __init__(self, module_type: type[Module]) -> None:
        self.module_type = module_type

    def setup_module(self, arguments: dict):
        self.torch_module = self.module_type(**arguments)
        self.setup_arguments = arguments
        self._input_shape = DimShape.from_module(self.torch_module)

        self.is_setup = True

    def get_arguments(self) -> dict: return self.setup_arguments

    def get_input_shape(self) -> tuple: return self._input_shape.sample()

    @staticmethod
    def from_type_str(type_str: str):
        match type_str.lower():
            case "linear":
                return ModulePart(nn.Linear)
            
            case "conv1d":
                return ModulePart(nn.Conv1d)
            
            case "conv2d":
                return ModulePart(nn.Conv2d)
            
            case "conv3d":
                return ModulePart(nn.Conv3d)
            
            case _:
                PART_LOGGER.error(f"ERROR: Couldn't parse type str '{type_str}'")
                return None
                # return ModulePart(nn.Module)