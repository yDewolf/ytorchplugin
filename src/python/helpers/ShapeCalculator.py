import logging
import torch
import torch.nn as nn
from data_types.ModuleParts import DIM_ANY_VALUE, DimShape, ModulePart

SHAPE_LOGGER = logging.Logger("ShapeLog")

class ShapeCalculator:
    @staticmethod
    def calculate_out_shape(module: ModulePart, test_shape: tuple | None = None) -> tuple:
        match module.module_type:
            # case nn.Linear:
            #     return ()
            
            # case nn.Conv2d:
            #     return ()

            case _:
                return ShapeCalculator._shape_calculation_fallback(module, test_shape)

    @staticmethod
    def _shape_calculation_fallback(module: ModulePart, test_shape: tuple | None) -> tuple:
        dtype = next(module.torch_module.parameters()).dtype
        device = next(module.torch_module.parameters()).device

        shape = module.get_input_shape() if test_shape is None else test_shape
        if not ShapeCalculator.check_module_input_shape(module, shape):
            return (-1, )

        test_tensor = torch.zeros(shape, dtype=dtype, device=device)
        with torch.no_grad():
            output: torch.Tensor = module.torch_module.forward(test_tensor)
            return output.shape
    
    @staticmethod
    def check_module_input_shape(module: ModulePart, shape: tuple) -> bool:
        is_compatible, wrong_dims = module._input_shape.is_compatible(shape)
        if not is_compatible:
            SHAPE_LOGGER.error(f"ERROR: Shape {shape} dimensions {wrong_dims} are not compatible with module shape: ({module._input_shape._make_shape_str()}) - Module Type: {module.module_type} - Module: {module.key if module.key != "" else module.get_arguments()}")
            return False
    
        return True
