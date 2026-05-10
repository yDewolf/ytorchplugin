from nodeserver.api.node.nodes import BaseNode
from nodeserver.wrapper.nodes.helpers.file.typing_file_reader import TypeFileReader, TypeReaderUtils, ConstructorModel
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror

from torchserver.nodeapi.nodes.base_torch_nodes import DatasetInputNode, TorchModuleNode, BaseTorchNode
from torchserver.nodeapi.nodes.dataset_nodes import MNISTDatasetNode
from torchserver.nodeapi.nodes.torch_nodes import LinearNode, ReshapeNode

DATASET_EXTENSIONS: list[str] = []
NODE_REGISTRY: dict[str, type[BaseNode]] = {
    "LinearNode": LinearNode,
    "RandomData": DatasetInputNode,
    "MNISTDataset": MNISTDatasetNode,
    "util_reshape": ReshapeNode
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

# _slot_types: dict[str, BaseSlotType] = {
#     "input": INPUT_TYPE,
#     "output": OUTPUT_TYPE
# }
# # TODO: Make a Tensor DataType or something like that
# _default_slots: dict[str, SlotData] = {
#     "in_0": SlotData(type="input", data_type=DataTypes.ARRAY),
#     "out_0": SlotData(type="output", data_type=DataTypes.ARRAY)
# }

TORCH_TYPES = TypeFileReader.new(0, "TorchTypes", slot_types, [])
TORCH_TYPES.set_new_constructors(TypeReaderUtils.make_constructors(
    TORCH_TYPES, {}, auto_parser, node_constructors
))
# TORCH_TYPES.set_new_constructors(TypeReaderUtils.make_constructors(
#     TORCH_TYPES, _default_slots, _torch_builder, TorchSlotOutput, [
#         ConstructorModel.new(
#             "DatasetInput", 
#             NodeData({"file": NodeFileParameter(extension_filter=DATASET_EXTENSIONS)}), 
#             {"out_0": SlotData(type="output", data_type=DataTypes.ARRAY),
#              "labels": SlotData(type="output", data_type=DataTypes.ARRAY)} # FIXME: Change this to a Tensor DataType 
#         ),
#         # FIXME Test:
#         ConstructorModel.new(
#             "MNISTDataset", 
#             NodeData({"file": NodeFileParameter(extension_filter=DATASET_EXTENSIONS)}), 
#             {"out_0": SlotData(type="output", data_type=DataTypes.ARRAY),
#              "labels": SlotData(type="output", data_type=DataTypes.ARRAY)} # FIXME: Change this to a Tensor DataType 
#         ),
#         ConstructorModel.new(
#             "Linear", 
#             NodeData({"out_features": NodeNumberParameter(type=DataTypes.UINT, range=[0, 1024], step=1)})),
#         ConstructorModel.new("Conv2d"),
#         ConstructorModel.new(
#             "util_reshape", 
#             NodeData({
#                 "auto": NodeNumberParameter(type=DataTypes.UINT, range=[0, 1], step=1),
#                 "shape": BaseNodeParameter(type=DataTypes.ARRAY)
#             })
#         ), # TODO: Add a NodeBooleanParameter
#     ]
# ))