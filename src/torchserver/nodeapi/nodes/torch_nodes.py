from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror, SlotMirror, SlotOutput

from torchserver.data_types.ModuleParts import ModulePart

class TorchUtilNode(BaseNode):
    def pre_forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]):
        pass

    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        output_data = {}
        return self.map_to_slots(output_data)
    
    def post_forward(self):
        pass

class TorchNode(BaseNode):
    module_part: ModulePart

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)

        module_part = ModulePart.from_type_str(self._mirror.type_name)
        if module_part:
            self.module_part = module_part
        
