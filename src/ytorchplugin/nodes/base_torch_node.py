from abc import ABC

from nodeserver.engine import BaseNode, NodeInputs, NodeOutputs
from pydantic import ConfigDict
import torch


class TorchInputModel(NodeInputs):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TorchOutputModel(NodeOutputs):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TorchNodeInput(TorchInputModel):
    in_0: torch.Tensor

class TorchNodeOutput(TorchOutputModel):
    out_0: torch.Tensor

class BaseTorchNode(BaseNode[TorchNodeInput, TorchNodeOutput], ABC):
    InputModel = TorchNodeInput
    OutputModel = TorchNodeOutput

