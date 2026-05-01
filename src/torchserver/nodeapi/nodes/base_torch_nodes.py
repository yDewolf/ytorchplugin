from torch import nn
from typing import Any, Optional

from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror, SlotMirror, SlotOutput
import torch

class TorchSlotOutput(SlotOutput):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: Any):
        if isinstance(new_value, torch.Tensor):
            if isinstance(self.value, torch.Tensor):
                if not new_value.equal(self._value):
                    self._value = new_value
                    self._version += 1
                return
            
            self._value = new_value
            self._version += 1
            return
        
        if self._value != new_value:
            self._value = new_value
            self._version += 1

class BaseTorchNode(BaseNode):
    # FIXME: lol
    def get_input_tensor(self, inputs: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> Optional[torch.Tensor]:
        for slot_mirror, connection_data in inputs.items():
            if slot_mirror.slot_name != "in_0": continue
            for source_slot, output in connection_data.items():
                if isinstance(output._value, torch.Tensor):
                    return output._value
        
        return None
    
    def _prepare_for_preview(self, tensor: torch.Tensor) -> list:
        with torch.no_grad():
            preview_tensor = tensor
            if tensor.dim() > 3:
                preview_tensor = tensor[0]

            t_min, t_max = preview_tensor.min(), preview_tensor.max()
            if t_max > t_min:
                preview_tensor = (preview_tensor - t_min) / (t_max - t_min) * 255
            
            return preview_tensor.byte().cpu().tolist()

class DatasetInputNode(BaseTorchNode):
    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.bypass_cache = True

    # Carregar o dataset caso ele não tenha sido carregado ainda
    def pre_forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]):
        pass

    # Selecionar um elemento aleatório do dataset
    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        output_data = {"out_0": torch.rand((64,))}
        return self.map_to_slots(output_data)
    
    # Fazer alguma limpeza se necessário (é interessante que o dataset ainda esteja carregado na memória para acelerar o pre_forward)
    def post_forward(self):
        pass

class TorchUtilNode(BaseTorchNode):
    # TODO: Implement some cool utility I guess
    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        tensor_data = self.get_input_tensor(input)
        return self.map_to_slots({
            "out_0": tensor_data
        })

class TorchModuleNode(BaseTorchNode):
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
        
        return params

    def _should_rebuild(self, new_params: dict) -> bool:
        return self.module is None or new_params != self._last_applied_params

    def build_torch_module(self, params: dict) -> nn.Module:
        raise NotImplementedError


    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        module_parameters = self._get_current_params(input)
        tensor_data = self.get_input_tensor(input)
        if self._should_rebuild(module_parameters):
            new_module = self.build_torch_module(module_parameters)
            new_module.to(self.device)
            
            # if self.module:
            #     new_module.load_state_dict(self.module.state_dict(), strict=False)
            
            self.module = new_module
            self._last_applied_params = module_parameters

        output_tensor: Optional[torch.Tensor] = None
        if self.module and isinstance(tensor_data, torch.Tensor):
            output_tensor = self.module(tensor_data)

        return self.map_to_slots({"out_0": output_tensor})
