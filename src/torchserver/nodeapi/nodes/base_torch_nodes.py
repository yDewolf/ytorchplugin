from collections import deque
import json
import os
import pathlib

from nodeserver.api.instance.instance_runtime import _ReadonlyContext, ContextAwareInput
from nodeserver.api.internal.instance_state import InternalNodeState
from nodeserver.api.node.abstract._nodes import _Node
from nodeserver.api.node.node_parameters import Param
from nodeserver.api.node.nodes import BaseNode, NoInput, NoOutput, NodeUtils
from nodeserver.wrapper.metadata.nodes.node_metadata import NodeTypeMeta
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
from pydantic import BaseModel
from torch import nn
from typing import Annotated, Any, Optional

import torch
from torchserver.nodeapi.nodes.metadata import BASE_NODE_TAG, TORCH_DATASET_CATEGORY, TORCH_LOSS_CATEGORY, TORCH_MODULE_CATEGORY, TORCH_NODE_CATEGORY, TORCH_TRAINER_CATEGORY, TORCH_UTIL_CATEGORY
from torchserver.nodeapi.nodes.torch_slots import TorchInput, TorchOutput

TORCH_MODEL_EXTENSION = "pt"

class TorchIOModel(BaseModel):
    class Config:
        arbitrary_types_allowed=True

class BaseTorchNode(BaseNode):
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_NODE_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

    def _prepare_for_preview(self, tensor: torch.Tensor) -> list:
        with torch.no_grad():
            preview_tensor = torch.tensor(tensor)
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
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_DATASET_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

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

    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_UTIL_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

class TorchModuleNode(BaseTorchNode):
    InputModel = _TorchModuleIn
    OutputModel = _TorchModuleOut

    module: Optional[nn.Module] = None
    device: torch.device = torch.device("cpu")

    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_MODULE_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.module: Optional[nn.Module] = None
        self._last_applied_params = {}

    def _get_current_params(self, inputs: _TorchModuleIn) -> dict:
        return {
            "input_shape": inputs.in_0.shape,
            **self._parameters.model_dump()
        }

    def _should_rebuild(self, new_params: dict) -> bool:
        return self.module is None or new_params != self._last_applied_params

    def build_torch_module(self, params: dict) -> nn.Module:
        raise NotImplementedError

    def pre_forward(self, input: _TorchModuleIn):
        module_parameters = self._get_current_params(input)
        if self._should_rebuild(module_parameters):
            # TODO: add some logger here
            print(f"Rebuilding module for node: {self}")
            new_module = self.build_torch_module(module_parameters)
            new_module.to(self.device)

            self.module = new_module
            self._last_applied_params = module_parameters


    def forward(self, input: _TorchModuleIn) -> _TorchModuleOut:
        tensor_data = input.in_0

        output_tensor: Optional[torch.Tensor] = None
        if self.module and isinstance(tensor_data, torch.Tensor):
            output_tensor = self.module(tensor_data)

        if output_tensor is None:
            raise Exception(f"Couldn't run module for node {self}")

        return _TorchModuleOut(
            out_0=output_tensor
        )


    def get_parameters(self) -> list[torch.nn.Parameter]:
        if self.module is None: return []

        return list(self.module.parameters())

    def save_state(self, root_state_path: str) -> InternalNodeState | None:
        if not self.module: 
            return
        
        file_path, filename = NodeUtils.make_state_file_path(self._mirror, root_state_path, TORCH_MODEL_EXTENSION)
        model_context = {}
        model_context["model_state"] = self.module.state_dict()
        model_context["module_parameters"] = self._last_applied_params
        torch.save(model_context, file_path)

        return InternalNodeState(
            relative_state_path=str(pathlib.Path(file_path).relative_to(root_state_path)),
            state_data=None
        )

    def load_state(self, root_state_path: str, state: InternalNodeState):
        if state.relative_state_path:
            state_file_path = os.path.join(root_state_path, state.relative_state_path)
            
            if state_file_path.endswith(TORCH_MODEL_EXTENSION):
                model_data = torch.load(state_file_path)

                self._last_applied_params = model_data["module_parameters"]
                self.module = self.build_torch_module(self._last_applied_params)
                self.module.to(self.device)

                self.module.load_state_dict(model_data["model_state"], strict=True)

class _LossNodeIn(TorchIOModel):
    preds: Annotated[torch.Tensor, TorchInput()]
    targets: Annotated[torch.Tensor, TorchInput()]
class _LossNodeOut(TorchIOModel):
    loss: Annotated[torch.Tensor, TorchOutput()]

class LossNode(BaseTorchNode):
    InputModel = _LossNodeIn
    OutputModel = _LossNodeOut

    class Parameters(BaseTorchNode.Parameters):
        criterion: Annotated[str, Param(
            label="Criterion",
            options=["mse", "bce", "cel"]
        )] = "mse"
        weight: Annotated[float, Param(
            label="Weight",
            min=0.01,
            max=1.0,
            step=0.05
        )] = 1.0

    _parameters: Parameters # type: ignore
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_LOSS_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

    criterion: Optional[nn.modules.loss._Loss] = None

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)

    def pre_forward(self, input: _LossNodeIn):
        match self._parameters.criterion:
            case "mse": self.criterion = nn.MSELoss()
            case "bce": self.criterion = nn.BCELoss()
            case "cel": self.criterion = nn.CrossEntropyLoss()


    def forward(self, input: _LossNodeIn) -> _LossNodeOut:
        if not self.criterion:
            raise Exception("Missing criterion for loss node")
        
        loss = self.criterion(input.preds, input.targets)
        return _LossNodeOut(
            loss=loss
        )
    
    def _prepare_for_preview(self, tensor: torch.Tensor) -> list:
        with torch.no_grad():
            preview_tensor = torch.tensor(tensor)
            return preview_tensor.cpu().tolist()

class _TrainerNodeInput(ContextAwareInput):
    loss: Annotated[torch.Tensor, TorchInput()]

class TrainerNode(BaseTorchNode):
    InputModel = _TrainerNodeInput

    class Parameters(BaseTorchNode.Parameters):
        lr: Annotated[float, Param(
            label="Learning Rate",
            min=0.0001,
            max=1.0,
            step=0.0001
        )] = 0.0001
    _parameters: Parameters # type: ignore
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_TRAINER_CATEGORY,
        capitalized_name="",
        tags=[BASE_NODE_TAG]
    )

    optimizer: Optional[torch.optim.Optimizer]
    _cached_params_ids: set[int]

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.bypass_cache = True
        self.optimizer = None
        self._cached_params_ids = set()

    def pre_forward(self, input: _TrainerNodeInput):
        if not input._context: return
        model_nodes = self.get_model_nodes(input._context)
        self._update_optimizer(model_nodes)
        return super().pre_forward(input)

    def forward(self, input: _TrainerNodeInput) -> NoOutput:
        if self.optimizer is None: return NoOutput()
        
        self.optimizer.zero_grad()
        input.loss.backward()
        self.optimizer.step()

        return NoOutput()

    def _update_optimizer(self, model_nodes: list[TorchModuleNode]):
        all_params = []
        current_params_ids = set()
        
        for node in model_nodes:
            params = node.get_parameters()
            all_params.extend(params)
            current_params_ids.update(id(p) for p in params)

        # Se a lista de parâmetros mudou (nó novo ou removido), recria o optimizer
        if current_params_ids != self._cached_params_ids:
            self.optimizer = torch.optim.AdamW(all_params, lr=self._parameters.lr)
            self._cached_params_ids = current_params_ids
            return
        
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._parameters.lr

    def get_model_nodes(self, context: _ReadonlyContext) -> list[TorchModuleNode]:
        if not context._scene:
            return []
        
        found_modules = []
        visited = set()
        queue: deque[NodeMirror] = deque()
        for slot in self._mirror.inputs:
            for connection in slot.connections:
                queue.append(connection.parent_node)

        while queue:
            node_mirror = queue.popleft()
            if node_mirror.uid in visited:
                continue
            
            visited.add(node_mirror.uid)
            node_instance = context._scene.get_node(node_mirror.uid)
            if isinstance(node_instance, TorchModuleNode):
                found_modules.append(node_instance)

            for input_slot in node_mirror.inputs:
                for connection in input_slot.connections:
                    queue.append(connection.parent_node)

        print(found_modules)
        return found_modules