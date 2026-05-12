from typing import Annotated

from nodeserver.api.node.node_parameters import Param
from nodeserver.api.node.nodes import BaseNode
from torch import nn
from torch.nn.modules import Module

from torchserver.nodeapi.nodes.base_torch_nodes import _TorchModuleIn, _TorchModuleOut, TorchModuleNode, TorchUtilNode

class LinearNode(TorchModuleNode):
    class Parameters(TorchModuleNode.Parameters):
        out_features: Annotated[int, Param(
            label="OutFeatures"
        )] = 1
    _parameters: Parameters # type: ignore

    def build_torch_module(self, params: dict) -> Module:
        input_shape = params.get("input_shape", (1, ))
        return nn.Linear(
            in_features=int(input_shape[-1]),
            out_features=self._parameters.out_features
        )


class Conv2DNode(TorchModuleNode):
    class Parameters(TorchModuleNode.Parameters):
        out_channels: Annotated[int, Param(
            label="OutFeatures"
        )] = 1
        kernel_size: Annotated[str, Param(
            label="KernelSize"
        )] = "3,3"
        padding: Annotated[str, Param(
            label="Padding"
        )] = "3,3"

    _parameters: Parameters # type: ignore

    def build_torch_module(self, params: dict) -> Module:
        parsed_kernel = [int(dim_size) for dim_size in self._parameters.kernel_size.split(",")]
        kernel_size = (parsed_kernel[0], parsed_kernel[1])
        
        parsed_padding = [int(dim_size) for dim_size in self._parameters.padding.split(",")]
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
            out_channels=self._parameters.out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

class ReshapeNode(TorchUtilNode):
    class Parameters(BaseNode.Parameters):
        shape: Annotated[str, Param(
            label="Shape",
        )] = ""
    _parameters: Parameters # type: ignore

    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        tensor_data = input.in_0
        
        target_shape = self._parameters.shape
        parsed_shape: tuple = ()
 
        if type(target_shape) is str:
            parsed_shape = tuple([int(dim_size) for dim_size in target_shape.split(",")])

        reshaped_tensor = tensor_data
        if len(parsed_shape) > 0:
            reshaped_tensor = tensor_data.reshape(parsed_shape)
        
        return _TorchModuleOut(
            out_0=reshaped_tensor
        )

class ArgmaxNode(TorchUtilNode):
    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        tensor_data = input.in_0
        argmax_tensor = tensor_data.argmax(keepdim=True) # TODO: Add parameter for keepdim
        
        return _TorchModuleOut(
            out_0=argmax_tensor
        )

