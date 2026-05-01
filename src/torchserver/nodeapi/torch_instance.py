from nodeserver.api.instance.server_instance import ServerInstance
from nodeserver.wrapper.nodes.node.base_nodes import SlotMirror, SlotOutput
import torch

from torchserver.nodeapi.internal.torch_scene import TorchNodeScene
from torchserver.nodeapi.nodes.torch_types import TORCH_TYPES
from torchserver.nodeapi.torch_server import TorchRuntime

class TorchServerInstance(ServerInstance): 
    def __init__(self, _default_types):
        # TODO Generate Types File
        types = TORCH_TYPES
        super().__init__(types)

        self._runtime = TorchRuntime()
        self._scene = TorchNodeScene([], self.mirror_manager)
    
    def _prepare_to_send_output(self, slot_results: dict[SlotMirror, SlotOutput] | None):
        result_data = {}
        if slot_results != None:
            for slot, result in slot_results.items():
                if isinstance(result._value, torch.Tensor):
                    with torch.no_grad():
                        result_data[slot.slot_name] = result._value.detach().numpy().tolist()
                        continue

                result_data[slot.slot_name] = result._value
        
        return result_data