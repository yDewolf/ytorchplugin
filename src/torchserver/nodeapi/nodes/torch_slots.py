from typing import Any, Generic, Optional

from nodeserver.api.node.abstract._slots import _SlotIO
from nodeserver.api.node.slots import BaseNodeType, NodeSlot, SlotConfig, T_SlotIO
from nodeserver.wrapper.nodes.data.node_data_types import ARRAY_TYPE
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

class TorchInputIO[inputType: Any](_TorchSlotIO[inputType, None]):
    _is_input = True

class TorchOutputIO[valueType: Any](_TorchSlotIO[None, valueType]):
    _is_input = False

def TorchInput(max_inputs: int = 1, datatype_override: Optional[BaseNodeType] = ARRAY_TYPE, **kwargs):
    return SlotConfig(slot_class=NodeSlot, input_io=TorchInputIO, output_io=TorchOutputIO, is_input=True, datatype_override=datatype_override, max_inputs=max_inputs, **kwargs)

def TorchOutput(datatype_override: Optional[BaseNodeType] = ARRAY_TYPE, **kwargs):
    return SlotConfig(slot_class=NodeSlot, input_io=TorchInputIO, output_io=TorchOutputIO, is_input=False, datatype_override=datatype_override, **kwargs)
