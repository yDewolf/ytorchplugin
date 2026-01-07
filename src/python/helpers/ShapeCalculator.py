import logging
import torch
import torch.nn as nn
from data_types.ModuleParts import DIM_ANY_VALUE, ModulePart

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
        is_compatible, wrong_dims = module._input_shape.is_compatible(shape)
        if not is_compatible:
            example_shape = module._input_shape.shape
            shape_str: str = ""
            for idx, value in enumerate(example_shape):
                if value == DIM_ANY_VALUE:
                    shape_str += f"[{module._input_shape.size_range[0]}, {module._input_shape.size_range[1]}]"                   
                else:
                    shape_str += str(value)

                if idx < len(example_shape) - 1:
                    shape_str += ", "

            SHAPE_LOGGER.error(f"ERROR: Shape {shape} dimensions {wrong_dims} are not compatible with module shape: ({shape_str}) - Module Type: {module.module_type} - Module Parameters: {module.get_arguments()}")
            return (-1, )

        test_tensor = torch.zeros(shape, dtype=dtype, device=device)
        with torch.no_grad():
            output: torch.Tensor = module.torch_module.forward(test_tensor)
            return output.shape