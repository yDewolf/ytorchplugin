from abc import ABC, abstractmethod
from typing import Optional

from nodeblocks.engine import LogicNodeState, ISceneStateProvider
from nodeblocks.protocols import NodeSceneData
from torch import nn
import torch

from ytorchplugin.nodes.base_torch_node import BaseTorchNode, TorchNodeInput, TorchNodeOutput

TORCH_MODEL_EXTENSION = "pt"

class TorchModuleNode(BaseTorchNode, ABC):
    module: Optional[nn.Module]
    device: torch.device = torch.device("cpu")

    def __init__(self, scene_data: NodeSceneData) -> None:
        super().__init__(scene_data)
        self.module: Optional[nn.Module] = None
        self._last_applied_params = {}

    # -- BaseNode methods

    def pre_forward(self, inputs: TorchNodeInput):
        module_parameters = self._get_current_params(inputs)
        if self._should_rebuild(module_parameters):
            # TODO: add some logger here
            print(f"Rebuilding module for node: {self}")
            new_module = self.build_torch_module(module_parameters)
            new_module.to(self.device)

            self.module = new_module
            self._last_applied_params = module_parameters


    def forward(self, inputs: TorchNodeInput) -> TorchNodeOutput:
        tensor_data = inputs.in_0

        output_tensor: Optional[torch.Tensor] = None
        if self.module and isinstance(tensor_data, torch.Tensor):
            output_tensor = self.module(tensor_data)

        if output_tensor is None:
            # TODO: use node exception here
            raise Exception(f"Couldn't run module for node {self}")

        return TorchNodeOutput(
            out_0=output_tensor
        )

    # -- Torch Stuff

    @abstractmethod
    def build_torch_module(self, params: dict) -> nn.Module:
        raise NotImplementedError

    def get_torch_module(self) -> nn.Module:
        if not self.module:
            raise Exception(f"Module not initialized for node {self.scene_data.uid}")
        
        return self.module
    

    def _get_current_params(self, inputs: TorchNodeInput):
        return {
            "input_shape": inputs.in_0.shape,
            **self.params.model_dump()
        }

    def _should_rebuild(self, new_params: dict) -> bool:
        return self.module is None or new_params != self._last_applied_params

    def get_parameters(self) -> list[torch.nn.Parameter]:
        if self.module is None: return []

        return list(self.module.parameters())

    # -- Save Stuff

    def save_state(self, state_provider: ISceneStateProvider) -> Optional[LogicNodeState]:
        if not self.module: 
            return

        folder = state_provider.get_node_state_folder(self.scene_data.uid)
        folder.mkdir(exist_ok=True)
        
        file_path = folder / f"module_weights.{TORCH_MODEL_EXTENSION}"
        
        model_context = {}
        model_context["model_state"] = self.module.state_dict()
        model_context["module_parameters"] = self._last_applied_params
        torch.save(model_context, file_path)

        return LogicNodeState(
            extra_files={"module_weights": file_path}
        )

    def load_state(self, state: LogicNodeState):
        if not state.extra_files: return

        weight_path = state.extra_files.get("module_weights")
        if not weight_path: return

        model_data = torch.load(weight_path)

        self._last_applied_params = model_data["module_parameters"]
        self.module = self.build_torch_module(self._last_applied_params)
        self.module.to(self.device)

        self.module.load_state_dict(model_data["model_state"], strict=True)


