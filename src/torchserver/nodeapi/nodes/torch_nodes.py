from typing import Annotated, Optional

from nodeserver.api.node.node_parameters import Param
from nodeserver.api.node.nodes import BaseNode
from torch import nn
import torch
from torch.nn.modules import Module

from torchserver.nodeapi.nodes.base_torch_nodes import _TorchModuleIn, _TorchModuleOut, TorchModuleNode, TorchUtilNode
from torchserver.nodeapi.nodes.torch_slots import TorchInput
from torchserver.nodeapi.utils.parameter_utils import ParameterUtils

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
        parsed_kernel = ParameterUtils.str_shape_to_tuple(self._parameters.kernel_size)
        kernel_size = (parsed_kernel[0], parsed_kernel[1])
        
        parsed_padding = ParameterUtils.str_shape_to_tuple(self._parameters.padding)
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
        parsed_shape = ParameterUtils.str_shape_to_tuple(self._parameters.shape)
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

class RepeatNode(TorchUtilNode):
    class Parameters(BaseNode.Parameters):
        shape: Annotated[str, Param(
            label="Shape",
        )] = ""
    _parameters: Parameters # type: ignore

    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        tensor_data = input.in_0
        parsed_shape = ParameterUtils.str_shape_to_tuple(self._parameters.shape)
        
        repeated_tensor = tensor_data
        if len(parsed_shape) > 0:
            repeated_tensor = tensor_data.repeat(parsed_shape)
        
        return _TorchModuleOut(
            out_0=repeated_tensor
        )

class _DoubleTensorIn(_TorchModuleIn):
    in_1: Annotated[torch.Tensor, TorchInput()]


class ConcatNode(TorchUtilNode):
    InputModel = _DoubleTensorIn
    class Parameters(BaseNode.Parameters):
        dim: Annotated[int, Param(
            label="Dimension",
        )] = 0
    _parameters: Parameters # type: ignore

    def forward(self, input: _DoubleTensorIn) -> _TorchModuleOut:
        target_dim = self._parameters.dim if self._parameters.dim else 0

        # shape_0 = [dim for idx, dim in enumerate(input.in_0.shape) if idx != target_dim]
        # shape_1 = [dim for idx, dim in enumerate(input.in_1.shape) if idx != target_dim]
        # if shape_0 != shape_1:
        #     raise Exception("Wrong Input Shapes")

        catted_tensor = torch.cat([input.in_0, input.in_1], dim=target_dim)

        return _TorchModuleOut(
            out_0=catted_tensor
        )

class StackNode(TorchUtilNode):
    InputModel = _DoubleTensorIn
    class Parameters(BaseNode.Parameters):
        dim: Annotated[int, Param(
            label="Dimension",
        )] = 0
    _parameters: Parameters # type: ignore

    def forward(self, input: _DoubleTensorIn) -> _TorchModuleOut:
        # if input.in_0.size() != input.in_1.size():
        #     raise Exception("Tensors should have the same size")

        target_dim = self._parameters.dim if self._parameters.dim else 0
        catted_tensor = torch.stack([input.in_0, input.in_1], dim=target_dim)

        return _TorchModuleOut(
            out_0=catted_tensor
        )


class SumNode(TorchUtilNode):
    InputModel = _DoubleTensorIn
    class Parameters(BaseNode.Parameters):
        weight_0: Annotated[float, TorchInput(
            label="Weight_0"
        )] = 1.0
        weight_1: Annotated[float, TorchInput(
            label="Weight_1"
        )] = 1.0
    _parameters: Parameters # type: ignore

    def forward(self, input: _DoubleTensorIn) -> _TorchModuleOut:
        # if input.in_0.size() != input.in_1.size():
        #     raise Exception("Tensors should have the same size")

        weight_0 = self._parameters.weight_0 if self._parameters.weight_0 else 1.0
        weight_1 = self._parameters.weight_1 if self._parameters.weight_1 else 1.0
        sum_tensor = input.in_0 * weight_0 + input.in_1 * weight_1
        return _TorchModuleOut(
            out_0=sum_tensor
        )

class NoGradNode(TorchUtilNode):
    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        with torch.no_grad():
            return _TorchModuleOut(
                out_0=input.in_0.detach()
            )