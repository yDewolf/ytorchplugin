from abc import ABC
from typing import Annotated

from torchserver.nodeapi.utils.parameter_utils import ParameterUtils
from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.parameters.param_wrappers import Param

from ytorchplugin.nodes.base_torch_node import BaseTorchNode, TorchNodeInput, TorchNodeOutput
from ytorchplugin.nodes.utils.util_nodes import TorchUtilNode

class TorchShapeUtilNode(BaseTorchNode, ABC):
    class Parameters(TorchUtilNode.Parameters):
        shape: Annotated[str, Param(required=True)]

    params: Parameters # type: ignore


@plugin_node(id="reshape")
class ReshapeNode(TorchShapeUtilNode):
    def forward(self, inputs: TorchNodeInput) -> TorchNodeOutput:
        tensor_data = inputs.in_0
        parsed_shape = ParameterUtils.str_shape_to_tuple(self.params.shape)
        reshaped_tensor = tensor_data
        if len(parsed_shape) > 0:
            reshaped_tensor = tensor_data.reshape(parsed_shape)

        return TorchNodeOutput(
            out_0=reshaped_tensor
        )


@plugin_node(id="repeat")
class RepeatNode(TorchShapeUtilNode):
    def forward(self, inputs: TorchNodeInput) -> TorchNodeOutput:
        tensor_data = inputs.in_0
        parsed_shape = ParameterUtils.str_shape_to_tuple(self.params.shape)

        repeated_tensor = tensor_data
        if len(parsed_shape) > 0:
            repeated_tensor = tensor_data.repeat(parsed_shape)

        return TorchNodeOutput(
            out_0=repeated_tensor
        )

