from typing import Annotated, Optional

from nodeserver.wrapper.metadata.nodes.node_metadata import NodeTypeMeta
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
from torch import nn
import torch

from torchserver.nodeapi.nodes.base_torch_nodes import _LossNodeIn, _LossNodeOut, BaseTorchNode, LossNode
from torchserver.nodeapi.nodes.metadata import TORCH_LOSS_CATEGORY
from torchserver.nodeapi.nodes.torch_slots import TorchOutput


class _CELNodeOut(_LossNodeOut):
    model_output: Annotated[torch.Tensor, TorchOutput()]

class CrossEntropyLossNode(LossNode):
    InputModel = _LossNodeIn
    OutputModel = _CELNodeOut

    class Parameters(BaseTorchNode.Parameters): # type: ignore
        pass
    
    _parameters: Parameters # type: ignore
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_LOSS_CATEGORY,
        capitalized_name="CrossEntropyLoss",
        tags=[]
    )

    def __init__(self, mirror: NodeMirror | None = None):
        self.criterion = nn.CrossEntropyLoss()
        super().__init__(mirror)

    def pre_forward(self, input: _LossNodeIn):
        pass

    def forward(self, input: _LossNodeIn) -> _CELNodeOut:
        if not self.criterion:
            raise Exception("Missing criterion for loss node")
        
        loss = self.criterion(input.preds, input.targets)
        class_idx = input.preds.argmax(keepdim=True)
        return _CELNodeOut(
            loss=loss,
            model_output=class_idx
        )
    