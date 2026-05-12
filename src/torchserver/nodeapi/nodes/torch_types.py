from nodeserver.api.node.nodes import BaseNode
from nodeserver.wrapper.nodes.helpers.file.typing_file_reader import TypeFileReader, TypeReaderUtils, ConstructorModel
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror

from torchserver.nodeapi.nodes.base_torch_nodes import DatasetInputNode, LossNode, TorchModuleNode, BaseTorchNode, TrainerNode
from torchserver.nodeapi.nodes.dataset_nodes import MNISTDatasetNode
from torchserver.nodeapi.nodes.torch_nodes import ArgmaxNode, Conv2DNode, LinearNode, ReshapeNode

DATASET_EXTENSIONS: list[str] = []
NODE_REGISTRY: dict[str, type[BaseNode]] = {
    "LinearNode": LinearNode,
    "Conv2D": Conv2DNode,
    "RandomData": DatasetInputNode,
    "MNISTDataset": MNISTDatasetNode,
    "util_reshape": ReshapeNode,
    "util_argmax": ArgmaxNode,
    "LossNode": LossNode,
    "TrainerNode": TrainerNode
}

node_constructors: list[ConstructorModel] = []
slot_types = {}
for node_type in NODE_REGISTRY:
    super_slot_types, constructor = NODE_REGISTRY[node_type].generate_types(slot_types, type_name=node_type)
    node_constructors.append(constructor)
    slot_types = super_slot_types

def auto_parser(mirror: NodeMirror) -> BaseNode:
    node_class = NODE_REGISTRY.get(mirror.type_name, None)
    if not node_class:
        raise Exception(f"Couldn't parse node with type {mirror.type_name}")

    return node_class(mirror)

TORCH_TYPES = TypeFileReader.new(0, "TorchTypes", slot_types, [])
TORCH_TYPES.set_new_constructors(TypeReaderUtils.make_constructors(
    TORCH_TYPES, {}, auto_parser, node_constructors
))
