from typing import Annotated

from nodeserver.engine.plugins.api.decorators import plugin_node
from nodeserver.engine.protocols import Param
from torch import nn

from ytorchplugin.nodes.torch_module_node import TorchModuleNode

@plugin_node(id="linear")
class LinearNode(TorchModuleNode):
    class Parameters(TorchModuleNode.Parameters):
        out_features: Annotated[int, Param(
            label="OutFeatures"
        )]

    params: Parameters # type: ignore

    def build_torch_module(self, params: dict) -> nn.Module:
        input_shape = params.get("input_shape", (1, ))
        return nn.Linear(
            in_features=int(input_shape[-1]),
            out_features=self.params.out_features
        )
