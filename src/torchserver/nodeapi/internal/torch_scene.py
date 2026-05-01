from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.api.instance.node_scene import NodeScene, logger
from nodeserver.wrapper.nodes.helpers.scene_manager import MirrorSceneManager
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
import torch

from torchserver.nodeapi.nodes.torch_nodes import TorchNode

class TorchNodeScene(NodeScene):
    def __init__(self, nodes: list[BaseNode], mirror_manager: MirrorSceneManager, device: str = "cpu"):
        super().__init__(nodes, mirror_manager)
        self.device = torch.device(device)
        self._node_instances: dict[str, BaseNode] = {}

    def update_nodes(self) -> bool:
        current_uids = set(self.mirror_manager.node_manager._nodes.keys())
        existing_uids = set(self._node_instances.keys())

        for uid in existing_uids - current_uids:
            del self._node_instances[uid]
        
        for uid, mirror in self.mirror_manager.node_manager._nodes.items():
            if uid not in self._node_instances:
                new_node = self.build_node(mirror)
                if new_node:
                    if isinstance(new_node, TorchNode):
                        new_node.device = self.device
                    self._node_instances[uid] = new_node
            else:
                self._node_instances[uid]._mirror = mirror

        self.nodes = list(self._node_instances.values())
        return True

    def to(self, device: str):
        self.device = torch.device(device)
        for node in self.nodes:
            if isinstance(node, TorchNode) and node.module:
                node.module.to(self.device)
                node.device = self.device