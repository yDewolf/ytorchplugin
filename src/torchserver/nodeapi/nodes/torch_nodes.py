from nodeserver.wrapper.nodes.node.base_nodes import SlotMirror, SlotOutput
from torch import nn
from torch.nn.modules import Module

from torchserver.nodeapi.nodes.base_torch_nodes import TorchModuleNode, TorchUtilNode


class LinearNode(TorchModuleNode):
    def build_torch_module(self, params: dict) -> Module:
        return nn.Linear(
            in_features=int(params.get("in_features", 1)),
            out_features=int(params.get("out_features", 1))
        )

class ReshapeNode(TorchUtilNode):
    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        tensor_data = self.get_input_tensor(input)
        if tensor_data == None:
            return self.map_to_slots({
                "out_0": tensor_data
            })
        
        target_shape = self._mirror.data.get_parameter_value("shape")
        parsed_shape: tuple = ()

        if type(target_shape) is str:
            parsed_shape = tuple([int(dim_size) for dim_size in target_shape.split(",")])

        reshaped_tensor = tensor_data
        if len(parsed_shape) > 0:
            reshaped_tensor = tensor_data.reshape(parsed_shape)
        
        return self.map_to_slots({
            "out_0": reshaped_tensor
        })