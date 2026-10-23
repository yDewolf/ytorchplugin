
from abc import ABC
from typing import Annotated

from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.node.logic_nodes import BaseNode
from nodeserver.engine.protocols.parameters.param_wrappers import Param
import torch

from ytorchplugin.nodes.base_torch_node import TorchNodeInput, TorchNodeOutput


class DoubleTensorIn(TorchNodeInput):
    in_1: torch.Tensor

class TensorOperationNode(BaseNode[DoubleTensorIn, TorchNodeOutput], ABC):
    InputModel = DoubleTensorIn
    OutputModel = TorchNodeOutput

class DimTensorOperationNode(TensorOperationNode, ABC):
    class Parameters(BaseNode.Parameters):
        dim: Annotated[int, Param(required=True)] = 0

    params: Parameters # type: ignore


@plugin_node(id="concat")
class ConcatNode(DimTensorOperationNode):
    def forward(self, inputs: DoubleTensorIn) -> TorchNodeOutput:
        target_dim = self.params.dim if self.params.dim else 0

        catted_tensor = torch.cat([inputs.in_0, inputs.in_1], dim=target_dim)

        return TorchNodeOutput(
            out_0=catted_tensor
        )

@plugin_node(id="stack")
class StackNode(DimTensorOperationNode):
    def forward(self, inputs: DoubleTensorIn) -> TorchNodeOutput:
        target_dim = self.params.dim if self.params.dim else 0
        catted_tensor = torch.stack([inputs.in_0, inputs.in_1], dim=target_dim)

        return TorchNodeOutput(
            out_0=catted_tensor
        )



@plugin_node(id="sumtensors")
class SumNode(TensorOperationNode):
    class Parameters(TensorOperationNode.Parameters):
        weight_0: Annotated[float, Param()] = 1.0
        weight_1: Annotated[float, Param()] = 1.0
    
    params: Parameters # type: ignore

    def forward(self, inputs: DoubleTensorIn) -> TorchNodeOutput:
        weight_0 = self.params.weight_0 if self.params.weight_0 else 1.0
        weight_1 = self.params.weight_1 if self.params.weight_1 else 1.0
        sum_tensor = inputs.in_0 * weight_0 + inputs.in_1 * weight_1

        return TorchNodeOutput(
            out_0=sum_tensor
        )

