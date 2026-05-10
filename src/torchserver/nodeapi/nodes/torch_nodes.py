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
        return nn.Linear(
            in_features=int(params.get("in_features", 1)),
            out_features=self._parameters.out_features
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