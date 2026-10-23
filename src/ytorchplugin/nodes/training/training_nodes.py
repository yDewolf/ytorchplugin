from collections import deque
from typing import Annotated, Optional

from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.node.logic_nodes import BaseNode, NodeOutputs
from nodeserver.engine.protocols.node.node_scene import NodeScene
from nodeserver.engine.protocols.parameters.param_wrappers import Param
from nodeserver.engine.runtime.context_injectors import SceneAwareInput
from nodeserver.protocols.manifest.node.node_graph import NodeSceneData

from ytorchplugin.datatypes.torch_types import LossDataType
from ytorchplugin.nodes.base_torch_node import TorchInputModel
from ytorchplugin.nodes.torch_module_node import TorchModuleNode
import torch


class TrainerNodeInput(TorchInputModel, SceneAwareInput):
    loss: LossDataType

@plugin_node(id="trainer")
class TrainerNode(BaseNode[TrainerNodeInput, NodeOutputs]):
    InputModel = TrainerNodeInput

    class Parameters(BaseNode.Parameters):
        lr: Annotated[float, Param(
            label="Learning Rate",
            max=1.0,
            step=1e-6
        )] = 1e-6

    params: Parameters # type: ignore

    optimizer: Optional[torch.optim.Optimizer]
    _cached_params_ids: set[int]

    def __init__(self, scene_data: NodeSceneData) -> None:
        super().__init__(scene_data)
        self.config.bypass_cache = True
        self.optimizer = None
        self._cached_params_ids = set()

    def pre_forward(self, inputs: TrainerNodeInput) -> None:
        model_nodes = self.get_model_nodes(inputs.scene_context)
        self._update_optimizer(model_nodes)
        return super().pre_forward(inputs)


    def forward(self, inputs: TrainerNodeInput) -> NodeOutputs:
        if self.optimizer is None: return NodeOutputs()
        
        self.optimizer.zero_grad()
        inputs.loss.loss_tensor.backward()
        self.optimizer.step()

        return NodeOutputs()


    def _update_optimizer(self, model_nodes: list[TorchModuleNode]):
        all_params = []
        current_params_ids = set()
        
        for node in model_nodes:
            params = node.get_parameters()
            all_params.extend(params)
            current_params_ids.update(id(p) for p in params)

        if current_params_ids != self._cached_params_ids:
            self.optimizer = torch.optim.AdamW(all_params, lr=self.params.lr or 0.0001)
            self._cached_params_ids = current_params_ids
            return
        
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.params.lr


    def get_model_nodes(self, scene: NodeScene) -> list[TorchModuleNode]:
        found_modules = []
        visited = set()

        queue: deque[str] = deque()
        queue += scene.graph.get_upstream_node_ids(self.scene_data.uid)
        while queue:
            node_uid = queue.popleft()
            if node_uid in visited:
                continue
            
            visited.add(node_uid)
            logic_node = scene.get_logic_node(node_uid)
            if isinstance(logic_node, TorchModuleNode):
                found_modules.append(logic_node)

            queue += scene.graph.get_upstream_node_ids(node_uid)

        print(found_modules)
        return found_modules
