from typing import Any, Optional

from nodeserver.api.node.abstract._nodes import _Node
from nodeserver.api.node.abstract._slots import _SlotIO
from nodeserver.api.web.instance.special_instance import UserSession, WsServerInstance
from nodeserver.wrapper.nodes.node.base_nodes import SlotMirror
from nodeserver.api.web.requests.websocket_requests import SlotOutputWrapper
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

    def _parse_slot_result(self, result: _SlotIO, node: _Node) -> SlotOutputWrapper:
        result_output = result.value
        if isinstance(result._value, torch.Tensor) and isinstance(node, BaseTorchNode):
            result_output = node._prepare_for_preview(result._value)

        converted_meta: Optional[dict[str, Any]] = {} if result.value_meta != None else None
        if result.value_meta != None and converted_meta != None:
            for key, value in result.value_meta.items() :
                if isinstance(value, torch.Size):
                    converted_meta[key] = list(value)

                if isinstance(value, torch.Tensor):
                    if value.size() == 1 or len(value.shape) == 0:
                        converted_meta[key] = float(value.detach().cpu())
                    else:
                        converted_meta[key] = list(value.detach().cpu())

        
        return SlotOutputWrapper(
            value=result_output,
            value_meta=converted_meta
        )