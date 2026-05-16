from nodeserver.api.node.nodes import BaseNode
from nodeserver.wrapper.utils.type_reader_utils import TypeReaderUtils

from torchserver.nodeapi.nodes.base_torch_nodes import DatasetInputNode, LossNode, TrainerNode
from torchserver.nodeapi.nodes.dataset_nodes import MNISTDatasetNode
from torchserver.nodeapi.nodes.loss_nodes import CrossEntropyLossNode
from torchserver.nodeapi.nodes.torch_nodes import ArgmaxNode, ConcatNode, Conv2DNode, LinearNode, RepeatNode, ReshapeNode, StackNode

DATASET_EXTENSIONS: list[str] = []
NODE_REGISTRY: dict[str, type[BaseNode]] = {
    "LinearNode": LinearNode,
    "Conv2D": Conv2DNode,
    "RandomData": DatasetInputNode,
    "MNISTDataset": MNISTDatasetNode,
    "util_reshape": ReshapeNode,
    "util_argmax": ArgmaxNode,
    "util_cat": ConcatNode,
    "util_stack": StackNode,
    "util_repeat": RepeatNode,
    "LossNode": LossNode,
    "CELNode": CrossEntropyLossNode,
    "TrainerNode": TrainerNode
}

TORCH_TYPES = TypeReaderUtils.make_types_from_registry(0, "TorchTypes", NODE_REGISTRY)
pass