from abc import ABC, abstractmethod
from typing import Annotated, Optional

from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.node.logic_nodes import BaseNode
from nodeserver.engine.protocols.parameters.param_wrappers import Param
from nodeserver.protocols.manifest.node.node_graph import NodeSceneData
import torch
from torch import nn

from ytorchplugin.datatypes.torch_types import LossDataType
from ytorchplugin.nodes.base_torch_node import TorchInputModel, TorchOutputModel

class LossNodeInput(TorchInputModel):
    preds: torch.Tensor
    targets: torch.Tensor

class LossNodeOutput(TorchOutputModel):
    loss: LossDataType


class LossNode[inputType: LossNodeInput, outputType: LossNodeOutput](BaseNode[inputType, outputType], ABC):
    InputModel = LossNodeInput
    OutputModel = LossNodeOutput

    class Parameters(BaseNode.Parameters):
        weight: Annotated[float, Param()] = 1.0

    params: Parameters # type: ignore

    criterion: Optional[nn.modules.loss._Loss] = None

    @abstractmethod
    def on_setup_params(self):
        raise NotImplementedError("Loss Node must implement on_setup_params to initialize its criterion")

    def forward(self, inputs: inputType) -> outputType:
        if not self.criterion:
            raise Exception("Missing criterion for loss node")
        
        loss = self.criterion(inputs.preds, inputs.targets)
        return LossNodeOutput(
            loss=loss
        ) # type: ignore

class CELNodeOutput(LossNodeOutput):
    model_output: torch.Tensor

@plugin_node(id="loss-cel")
class CrossEntropyLossNode(LossNode[LossNodeInput, CELNodeOutput]):
    OutputModel = LossNodeOutput

    def on_setup_params(self):
        self.criterion = nn.CrossEntropyLoss(torch.tensor(self.params.weight))

    def forward(self, inputs: LossNodeInput) -> CELNodeOutput:
        if not self.criterion:
            raise Exception("Missing criterion for loss node")
        
        loss = self.criterion(inputs.preds, inputs.targets)
        class_idx = inputs.preds.argmax(keepdim=True)
        return CELNodeOutput(
            loss=loss,
            model_output=class_idx
        )

@plugin_node(id="loss-bce")
class BinaryCrossEntropyLossNode(LossNode[LossNodeInput, LossNodeOutput]):
    def on_setup_params(self):
        self.criterion = nn.BCELoss(torch.tensor(self.params.weight))

@plugin_node(id="loss-mse")
class MinimumSquaredErrorLossNode(LossNode[LossNodeInput, LossNodeOutput]):
    def on_setup_params(self):
        self.criterion = nn.MSELoss(torch.tensor(self.params.weight))
