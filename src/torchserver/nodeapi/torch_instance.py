from nodeserver.api.node.abstract._nodes import _Node
from nodeserver.api.node.abstract._slots import _SlotIO
from nodeserver.api.web.instance.special_instance import UserSession, WsServerInstance
from nodeserver.wrapper.nodes.node.base_nodes import SlotMirror
import torch

from torchserver.nodeapi.internal.torch_scene import TorchNodeScene
from torchserver.nodeapi.nodes.base_torch_nodes import BaseTorchNode
from torchserver.nodeapi.nodes.torch_types import TORCH_TYPES
from torchserver.nodeapi.torch_server import TorchRuntime

class TorchServerInstance(WsServerInstance): 
    def __init__(self, related_session: UserSession, _default_types):
        # TODO Generate Types File
        types = TORCH_TYPES
        super().__init__(related_session, types)

        self._runtime = TorchRuntime()
        self._scene = TorchNodeScene([], self.mirror_manager)
    
    def _prepare_to_send_output(self, node: _Node, slot_results: dict[SlotMirror, _SlotIO] | None):
        result_data = {}
        if slot_results != None:
            for slot, result in slot_results.items():
                if isinstance(result._value, torch.Tensor) and isinstance(node, BaseTorchNode):
                    result_data[slot.slot_id] = node._prepare_for_preview(result._value)
                    continue

                result_data[slot.slot_id] = result._value
        
        return result_data