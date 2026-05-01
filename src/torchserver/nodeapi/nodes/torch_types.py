from nodeserver.wrapper.nodes.data.node_data_types import INPUT_TYPE, OUTPUT_TYPE, BaseSlotType
from nodeserver.wrapper.nodes.helpers.file.type_dataclasses import DataTypes, NodeNumberParameter, SlotData
from nodeserver.wrapper.nodes.helpers.file.typing_file_reader import TypeFileReader, TypeReaderUtils, ConstructorModel
from nodeserver.api.instance.base_nodes import BaseNode
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
from nodeserver.wrapper.nodes.data.node_data import NodeData
from nodeserver.wrapper.nodes.helpers.file.type_dataclasses import NodeFileParameter

from torchserver.nodeapi.nodes.torch_nodes import TorchNode, TorchUtilNode

DATASET_EXTENSIONS: list[str] = []

def _torch_builder(mirror: NodeMirror) -> BaseNode:
    if mirror.type_name.startswith("util"):
        return TorchUtilNode(mirror)

    return TorchNode(mirror)

_slot_types: dict[str, BaseSlotType] = {
    "input": INPUT_TYPE,
    "output": OUTPUT_TYPE
}
# TODO: Make a Tensor DataType or something like that
_default_slots: dict[str, SlotData] = {
    "in_0": SlotData(type="input", data_type=DataTypes.ARRAY),
    "out_0": SlotData(type="output", data_type=DataTypes.ARRAY)
}

TORCH_TYPES = TypeFileReader.new(0, "TorchTypes", _slot_types, [])
TORCH_TYPES.set_new_constructors(TypeReaderUtils.make_constructors(
    TORCH_TYPES, _default_slots, _torch_builder, [
        ConstructorModel.new(
            "DatasetInput", 
            NodeData({"file": NodeFileParameter(extension_filter=DATASET_EXTENSIONS)}), 
            {"out_0": SlotData(type="output", data_type=DataTypes.ARRAY)} # FIXME: Change this to a Tensor DataType 
        ),
        ConstructorModel.new("Linear"),
        ConstructorModel.new("Conv2d"),
        ConstructorModel.new(
            "util_reshape", 
            NodeData({"auto": NodeNumberParameter(type=DataTypes.UINT, range=[0, 1], step=1)})
        ), # TODO: Add a NodeBooleanParameter
    ]
))