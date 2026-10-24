from abc import ABC, abstractmethod
from typing import Annotated

from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.node.logic_nodes import BaseNode, NodeInputs
from nodeserver.engine.protocols.parameters.node_parameter import NodeParameters
from nodeserver.engine.protocols.parameters.param_wrappers import Param
from nodeserver.protocols.manifest.node.node_graph import NodeSceneData
import torch

from ytorchplugin.nodes.base_torch_node import TorchOutputModel


class DatasetOut(TorchOutputModel):
    sample: torch.Tensor
    target: torch.Tensor

# @plugin_node(id="random_tensor")
class DatasetInputNode(BaseNode[NodeInputs, DatasetOut], ABC):
    OutputModel = DatasetOut

    class Parameters(NodeParameters): # type: ignore
        shape: Annotated[str, Param(required=True)]
    
    params: Parameters # type: ignore

    def __init__(self, scene_data: NodeSceneData):
        super().__init__(scene_data)
        self.config.bypass_cache = True

    # Carregar o dataset caso ele não tenha sido carregado ainda
    def pre_forward(self, inputs: NodeInputs):
        pass

    # Selecionar um elemento aleatório do dataset
    @abstractmethod
    def forward(self, inputs: NodeInputs) -> DatasetOut:
        pass
 
    # Fazer alguma limpeza se necessário (é interessante que o dataset ainda esteja carregado na memória para acelerar o pre_forward)
    def post_forward(self):
        pass

