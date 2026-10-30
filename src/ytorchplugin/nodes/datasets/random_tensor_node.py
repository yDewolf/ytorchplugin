from nodeserver.engine.plugins.api.decorators import plugin_node
from nodeserver.engine import NodeInputs
import torch

from ytorchplugin.nodes.datasets.base_dataset_node import DatasetInputNode, DatasetOut


@plugin_node(id="random_tensor")
class RandomNodeTensor(DatasetInputNode):
    def forward(self, inputs: NodeInputs) -> DatasetOut:
        parsed_shape = tuple([int(dim_size) for dim_size in self.params.shape.split(",")])
        
        torch.seed()
        random_tensor = torch.randn(parsed_shape)
        return DatasetOut(
            sample=random_tensor,
            target=torch.zeros(parsed_shape)
        )
