from abc import ABC

from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
import torch

from ytorchplugin.nodes.base_torch_node import BaseTorchNode, TorchNodeInput, TorchNodeOutput, TorchOutputModel


class TorchUtilNode(BaseTorchNode, ABC):
    pass

@plugin_node(id="argmax")
class ArgmaxNode(TorchUtilNode):
    def forward(self, inputs: TorchNodeInput) -> TorchNodeOutput:
        tensor_data = inputs.in_0
        argmax_tensor = tensor_data.argmax(keepdim=True) # TODO: Add parameter for keepdim
        
        return TorchNodeOutput(
            out_0=argmax_tensor
        )

@plugin_node(id="nograd")
class NoGradnode(TorchUtilNode):
    def forward(self, inputs: TorchNodeInput) -> TorchNodeOutput:
        with torch.no_grad():
            return TorchNodeOutput(
                out_0=inputs.in_0.detach()
            )

