from typing import Annotated, Optional

from nodeblocks.engine import NodeInputs, NodeParameters
from nodeblocks.engine.plugins.api.decorators import plugin_node
from nodeblocks.engine.protocols import Param
from nodeblocks.protocols import NodeSceneData

from ytorchplugin.nodes.datasets.base_dataset_node import DatasetInputNode, DatasetOut

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@plugin_node(id="dataset-mnist")
class MNISTDatasetNode(DatasetInputNode):
    dataloader: Optional[DataLoader]

    class Parameters(NodeParameters): # type: ignore
        batch_size: Annotated[int, Param(
            required=True,
            min=1
        )]

    params: Parameters # type: ignore
    def __init__(self, scene_data: NodeSceneData):
        super().__init__(scene_data)
        self.dataset = None
        self.dataloader = None
        self.data_iter = None
        self._last_path = None
        self._last_batch_size = None

    def _load_dataset(self, path: str, batch_size: int = 1):
        """Carrega o MNIST da memória/disco"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        self.data_iter = iter(self.dataloader)
        self._last_path = path
        self._last_batch_size = batch_size


    def forward(self, inputs: NodeInputs) -> DatasetOut:
        path = "./data"
        batch_size = self.params.batch_size
        if path != self._last_path or batch_size != self._last_batch_size:
            self._load_dataset(path, batch_size)

        try:
            samples, labels = next(self.data_iter)
        except (StopIteration, TypeError):
            self.data_iter = iter(self.dataloader)
            samples, labels = next(self.data_iter)

        return DatasetOut(
            sample=samples,
            target=labels.detach().clone()
        )


