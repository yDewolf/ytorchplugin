from typing import Annotated, Optional

from nodeserver.api.node.node_parameters import Param
from nodeserver.api.node.nodes import BaseNode
from nodeserver.wrapper.metadata.nodes.node_metadata import INPUT_CATEGORY, NodeTypeMeta
from nodeserver.wrapper.utils.type_reader_utils import TypeReaderUtils
from pydantic import BaseModel

from torchserver.nodeapi.nodes.activation.base_activation import ReLU_Node, Sigmoid_Node
from torchserver.nodeapi.nodes.base_torch_nodes import DatasetInputNode, LossNode, TrainerNode
from torchserver.nodeapi.nodes.dataset_nodes import MNISTDatasetNode, SeededRandomDataset
from torchserver.nodeapi.nodes.loss_nodes import CrossEntropyLossNode
from torchserver.nodeapi.nodes.torch_nodes import ArgmaxNode, ConcatNode, Conv2DNode, LinearNode, NoGradNode, RepeatNode, ReshapeNode, StackNode, SumNode

class _InputNodeOutput(BaseModel):
    out_0: Optional[int]

class MyInputNode(BaseNode):
    OutputModel = _InputNodeOutput
    class Parameters(BaseModel):
        value: Annotated[int, Param(
            label="Value",
            step=1
        )] = 0
    _parameters: Parameters

    def forward(self, input: BaseModel) -> _InputNodeOutput:
        value = self._parameters.value
        if value == None:
            # raise NoOutputException()
            return _InputNodeOutput(out_0=None)
        
        if type(value) != int:
            return _InputNodeOutput(out_0=None)
        
        return _InputNodeOutput(
            out_0=value
        )

    _metadata: NodeTypeMeta = NodeTypeMeta(
        category=INPUT_CATEGORY,
        capitalized_name="InputNode",
    )

DATASET_EXTENSIONS: list[str] = []
NODE_REGISTRY: dict[str, type[BaseNode]] = {
    "LinearNode": LinearNode,
    "Conv2D": Conv2DNode,
    "RandomData": DatasetInputNode,
    "SeededRandomData": SeededRandomDataset,
    "MNISTDataset": MNISTDatasetNode,
    "util_reshape": ReshapeNode,
    "util_argmax": ArgmaxNode,
    "util_cat": ConcatNode,
    "util_stack": StackNode,
    "util_repeat": RepeatNode,
    "torch_no_grad": NoGradNode,
    "math_sum": SumNode,
    "LossNode": LossNode,
    "CELNode": CrossEntropyLossNode,
    "TrainerNode": TrainerNode,
    "act_relu": ReLU_Node,
    "act_sigmoid": Sigmoid_Node,
    "math_input": MyInputNode,
}

TORCH_TYPES = TypeReaderUtils.make_types_from_registry(0, "TorchTypes", NODE_REGISTRY)
pass