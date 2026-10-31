from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node

from ytorchplugin.nodes.torch_module_node import TorchModuleNode
import torch.nn as nn

@plugin_node(id="actv_relu")
class ReLUNode(TorchModuleNode):
    def build_torch_module(self, params: dict) -> nn.Module:
        return nn.ReLU()

@plugin_node(id="actv_sigmoid")
class SigmoidNode(TorchModuleNode):
    def build_torch_module(self, params: dict) -> nn.Module:
        return nn.Sigmoid()

@plugin_node(id="actv_softmax")
class SoftmaxNode(TorchModuleNode):
    def build_torch_module(self, params: dict) -> nn.Module:
        return nn.Softmax()

