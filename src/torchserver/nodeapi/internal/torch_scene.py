from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.api.instance.node_scene import NodeScene, logger
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror

from torchserver.nodeapi.nodes.torch_nodes import TorchNode

# I may need to change some behavior here idk
class TorchModelScene(NodeScene):
    
    def build_node(self, mirror: NodeMirror) -> BaseNode | None:
        constructor = self.mirror_manager.type_reader.node_constructors.get(mirror.type_name)
        if not constructor:
            return None
    
        new_node = constructor.build_node(mirror)
        return new_node

    def update_nodes(self) -> bool:
        self.nodes.clear() # FIXME Update only nodes that changed
        for id, mirror in self.mirror_manager.node_manager._nodes.items():
            new_node = self.build_node(mirror)
            if not new_node:
                return False

            logger.debug(f"Parsed mirror to {new_node}")
            self.nodes.append(new_node)
        
        for node in self.nodes:
            if not isinstance(node, TorchNode):
                continue

            node.module_part.setup_module({})

        return True