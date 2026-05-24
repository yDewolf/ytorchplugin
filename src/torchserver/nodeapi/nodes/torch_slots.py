from typing import Any, Generic, Optional

from nodeserver.api.node.abstract._slots import _SlotIO
from nodeserver.api.node.slots import NodeSlot, SlotConfig
from nodeserver.wrapper.nodes.data.node_data_types import DefaultDataTypes
import torch


class _TorchSlotIO[inputType: Any, valueType: Any](_SlotIO[inputType, valueType]):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: valueType):
        if isinstance(new_value, torch.Tensor):
            if isinstance(self.value, torch.Tensor) and new_value.equal(self.value):
                return
        
        else:
            if self._value == new_value:
                return
        
        self._value = new_value # type: ignore
        self._version += 1

def TorchInput(max_inputs: int = 1, base_type_override: Optional[DefaultDataTypes] = DefaultDataTypes.ARRAY, **kwargs):
    return SlotConfig(slot_class=NodeSlot, slot_io=_TorchSlotIO, is_input=True, base_type_override=base_type_override, max_inputs=max_inputs, **kwargs)

def TorchOutput(base_type_override: Optional[DefaultDataTypes] = DefaultDataTypes.ARRAY, **kwargs):
    return SlotConfig(slot_class=NodeSlot, slot_io=_TorchSlotIO, is_input=False, base_type_override=base_type_override, **kwargs)
