from nodeserver.api.node.node_parameters import Param
from nodeserver.api.node.nodes import BaseNode, NoInput
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
from pydantic import BaseModel
from torch import nn
from typing import Annotated, Any, Optional

import torch

from torchserver.nodeapi.nodes.torch_slots import TorchInput, TorchOutput

class TorchIOModel(BaseModel):
    class Config:
        arbitrary_types_allowed=True

class BaseTorchNode(BaseNode):
    def _prepare_for_preview(self, tensor: torch.Tensor) -> list:
        with torch.no_grad():
            preview_tensor = tensor
            if tensor.dim() > 3:
                preview_tensor = tensor[0]

            t_min, t_max = preview_tensor.min(), preview_tensor.max()
            if t_max > t_min:
                preview_tensor = (preview_tensor - t_min) / (t_max - t_min) * 255
            
            return preview_tensor.byte().cpu().tolist()

class _DatasetOut(TorchIOModel):
    sample: Annotated[torch.Tensor, TorchOutput()]
    target: Annotated[torch.Tensor, TorchOutput()]

class DatasetInputNode(BaseTorchNode):
    OutputModel = _DatasetOut
    class Parameters(BaseNode.Parameters):
        shape: Annotated[str, Param(
            label="Shape",
        )] = "64,64"
    
    _parameters: Parameters # type: ignore

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.bypass_cache = True

    # Carregar o dataset caso ele não tenha sido carregado ainda
    def pre_forward(self, input: NoInput):
        pass

    # Selecionar um elemento aleatório do dataset
    def forward(self, input: NoInput) -> _DatasetOut:
        parsed_shape = tuple([int(dim_size) for dim_size in self._parameters.shape.split(",")])

        random_tensor = torch.randn(parsed_shape)
        return _DatasetOut(
            sample=random_tensor,
            target=random_tensor
        )

    # Fazer alguma limpeza se necessário (é interessante que o dataset ainda esteja carregado na memória para acelerar o pre_forward)
    def post_forward(self):
        pass


class _TorchModuleIn(TorchIOModel):
    in_0: Annotated[torch.Tensor, TorchInput()]
class _TorchModuleOut(TorchIOModel):
    out_0: Annotated[torch.Tensor, TorchOutput()]

class TorchUtilNode(BaseTorchNode):
    InputModel = _TorchModuleIn
    OutputModel = _TorchModuleOut

class TorchModuleNode(BaseTorchNode):
    InputModel = _TorchModuleIn
    OutputModel = _TorchModuleOut

    module: Optional[nn.Module] = None
    device: torch.device = torch.device("cpu")

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.module: Optional[nn.Module] = None
        self._last_applied_params = {}

    def _get_current_params(self, inputs: _TorchModuleIn) -> dict:
        return {
            "in_features": int(inputs.in_0.shape[-1])
        }

    def _should_rebuild(self, new_params: dict) -> bool:
        return self.module is None or new_params != self._last_applied_params

    def build_torch_module(self, params: dict) -> nn.Module:
        raise NotImplementedError


    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        module_parameters = self._get_current_params(input)
        tensor_data = input.in_0

        if self._should_rebuild(module_parameters):
            # TODO: add some logger here
            print(f"Rebuilding module for node: {self}")
            new_module = self.build_torch_module(module_parameters)
            new_module.to(self.device)

            self.module = new_module
            self._last_applied_params = module_parameters

        output_tensor: Optional[torch.Tensor] = None
        if self.module and isinstance(tensor_data, torch.Tensor):
            output_tensor = self.module(tensor_data)

        if output_tensor is None:
            raise Exception(f"Couldn't run module for node {self}")

        return _TorchModuleOut(
            out_0=output_tensor
        )