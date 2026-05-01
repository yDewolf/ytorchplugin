from torch import nn
from typing import Optional

from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror, SlotMirror, SlotOutput
import torch

from torchserver.data_types.ModuleParts import ModulePart

class DatasetInputNode(BaseNode):
    def pre_forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]):
        pass

    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        output_data = {"out_0": torch.zeros((64,))}
        return self.map_to_slots(output_data)
    
    def post_forward(self):
        pass

class TorchUtilNode(BaseNode):
    def pre_forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]):
        pass

    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        output_data = {}
        return self.map_to_slots(output_data)
    
    def post_forward(self):
        pass

class TorchNode(BaseNode):
    module: Optional[nn.Module] = None
    device: torch.device = torch.device("cpu")

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.module: Optional[nn.Module] = None
        self._last_applied_params = {}

    def _get_current_params(self, inputs: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict:
        params = {}
        for param_name, parameter in self._mirror.data.parameters.items():
            params[param_name] = parameter.value

        for slot_mirror, connection_data in inputs.items():
            if slot_mirror.slot_name != "in_0": continue

            # FIXME
            for source_slot, output in connection_data.items():
                if isinstance(output._value, torch.Tensor):
                    params["in_features"] = int(output._value.shape[-1])
                    
                params["tensor"] = output._value
        
        return params

    def _should_rebuild(self, new_params: dict) -> bool:
        # Verifica se os parâmetros que afetam a estrutura (ex: in_features) mudaram
        return self.module is None or new_params != self._last_applied_params

    def build_torch_module(self, params: dict) -> nn.Module:
        part = ModulePart.from_type_str(self._mirror.type_name.lower())
        if part:
            part.setup_module({"in_features": params["in_features"], "out_features": params["out_features"]})
            return part.torch_module
        
        raise NotImplementedError


    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        params = self._get_current_params(input)

        if self._should_rebuild(params):
            new_module = self.build_torch_module(params)
            new_module.to(self.device)
            
            # if self.module:
            #     new_module.load_state_dict(self.module.state_dict(), strict=False)
            
            self.module = new_module
            self._last_applied_params = params

        tensor_data = params.get("tensor") 
        if self.module and isinstance(tensor_data, torch.Tensor):
            output_tensor = self.module(tensor_data)
        else:
            output_tensor = None # Ou erro/bypass

        return self.map_to_slots({"out_0": output_tensor})