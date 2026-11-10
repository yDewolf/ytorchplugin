from typing import Annotated

from nodeblocks.engine.plugins.api.decorators import plugin_node
from nodeblocks.engine.protocols import Param

from ytorchplugin.utils.parameter_utils import ParameterUtils
from ytorchplugin.nodes.torch_module_node import TorchModuleNode
from torch import nn

@plugin_node(id="conv2d")
class Conv2DNode(TorchModuleNode):
    class Parameters(TorchModuleNode.Parameters):
        out_channels: Annotated[int, Param(
            label="OutFeatures"
        )]
        kernel_size: Annotated[str, Param(
            label="KernelSize"
        )]
        padding: Annotated[str, Param(
            label="Padding"
        )]

    params: Parameters # type: ignore

    def build_torch_module(self, params: dict) -> nn.Module:
        parsed_kernel = ParameterUtils.str_shape_to_tuple(self.params.kernel_size)
        kernel_size = (parsed_kernel[0], parsed_kernel[1])
        
        parsed_padding = ParameterUtils.str_shape_to_tuple(self.params.padding)
        padding = (parsed_padding[0], parsed_padding[1])

        input_shape = params.get("input_shape", (1, 1, 1, 1))
        C = 1
        if len(input_shape) == 4:
            B, C, H, W = input_shape
        elif len(input_shape) == 3:
            C, H, W = input_shape
        else:
            raise Exception("Wrong input for Conv2D, should be 4D or 3D (B, C, H, W) or (C, H, W)")

        return nn.Conv2d(
            in_channels=C,
            out_channels=int(self.params.out_channels),
            kernel_size=kernel_size,
            padding=padding
        )
