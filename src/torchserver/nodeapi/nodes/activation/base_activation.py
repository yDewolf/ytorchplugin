
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror
from torch import nn
from torch.nn.modules import Module

from torchserver.nodeapi.nodes.base_torch_nodes import TorchModuleNode

class ReLU_Node(TorchModuleNode):
    def build_torch_module(self, params: dict) -> Module:
        return nn.ReLU()


class Sigmoid_Node(TorchModuleNode):
    def build_torch_module(self, params: dict) -> Module:
        return nn.Sigmoid()
