from collections import deque
from typing import Annotated

from nodeblocks.engine.plugins.api.decorators import plugin_node
from nodeblocks.engine.protocols import Param
from nodeblocks.engine import SceneGraph, BaseNode, NodeScene
from nodeblocks.engine.runtime import SceneAwareInput
from nodeblocks.protocols import NodeSceneData

from ytorchplugin.nodes.base_torch_node import TorchInputModel, TorchOutputModel
from ytorchplugin.nodes.torch_module_node import TorchModuleNode
from torchinfo import summary
from torch import nn
import torch


class TorchInfoNodeInput(TorchInputModel, SceneAwareInput):
    in_0: torch.Tensor

class TorchInfoNodeOutput(TorchOutputModel):
    summary: str


@plugin_node(id="torch_info")
class TorchInfoNode(BaseNode[TorchInfoNodeInput, TorchInfoNodeOutput]):
    InputModel = TorchInfoNodeInput
    OutputModel = TorchInfoNodeOutput

    class Parameters(BaseNode.Parameters):
        input_size_str: Annotated[str, Param(
            label="Input Size",
        )] = "1,1,28,28"

    params: Parameters  # type: ignore

    def __init__(self, scene_data: NodeSceneData) -> None:
        super().__init__(scene_data)
        self.config.bypass_cache = True  # Garante que recalcule ao atualizar o grafo

    def forward(self, inputs: TorchInfoNodeInput) -> TorchInfoNodeOutput:
        model_nodes = self.get_ordered_model_nodes(inputs.scene_context)
        
        if not model_nodes:
            raise Exception("Couldn't find any TorchModules")

        composite_model = SceneGraphModule(model_nodes, inputs.scene_context.graph)

        try:
            input_size = tuple(int(x.strip()) for x in self.params.input_size_str.split(","))
        except Exception as e:
            raise Exception(f"Failed to parse input_size {e}")

        try:
            model_summary = summary(composite_model, input_size=input_size, verbose=0)
            summary_str = str(model_summary)
        except Exception as e:
            print(e)
            raise Exception(f"Failed to generate model summary {e}")

        return TorchInfoNodeOutput(
            summary=summary_str
        )

    def get_ordered_model_nodes(self, scene: NodeScene) -> list[TorchModuleNode]:
        found_modules = []
        visited = set()

        upstream_nodes = scene.graph.get_upstream_node_ids(self.scene_data.uid)
        queue: deque[str] = deque(list(upstream_nodes))
        
        while queue:
            node_uid = queue.popleft()
            if node_uid in visited:
                continue
            
            visited.add(node_uid)
            logic_node = scene.get_logic_node(node_uid)
            
            if isinstance(logic_node, TorchModuleNode):
                found_modules.append(logic_node)

            for upstream_id in scene.graph.get_upstream_node_ids(node_uid):
                queue.append(upstream_id)
        
        found_modules.reverse()
        return found_modules


class SceneGraphModule(nn.Module):
    def __init__(self, ordered_nodes: list[TorchModuleNode], scene_graph: SceneGraph):
        super().__init__()
        self.node_modules = nn.ModuleDict()
        self.node_map = {}
        
        for node in ordered_nodes:
            uid = node.scene_data.uid
            inner_mod = node.get_torch_module()
            self.node_modules[uid] = inner_mod
            self.node_map[uid] = node

        self.scene_graph = scene_graph
        self.ordered_nodes = ordered_nodes

    def forward(self, x):
        tensor_cache = {}
        if self.ordered_nodes:
            first_uid = self.ordered_nodes[0].scene_data.uid
            tensor_cache[first_uid] = x

        for node in self.ordered_nodes:
            uid = node.scene_data.uid
            module = self.node_modules[uid]

            upstream_ids = self.scene_graph.get_upstream_node_ids(uid)
            
            if not upstream_ids:
                node_input = tensor_cache.get(uid, x)
            else:
                parent_outputs = [tensor_cache[p_id] for p_id in upstream_ids if p_id in tensor_cache]
                
                if len(parent_outputs) == 1:
                    node_input = parent_outputs[0]
                elif len(parent_outputs) > 1:
                    node_input = sum(parent_outputs) 
                else:
                    node_input = x

            try:
                out = module(node_input)
                tensor_cache[uid] = out
            except Exception as e:
                raise RuntimeError(f"Failed to process node: {uid} ({type(module).__name__}): {e}")

        last_uid = self.ordered_nodes[-1].scene_data.uid
        return tensor_cache.get(last_uid, x)