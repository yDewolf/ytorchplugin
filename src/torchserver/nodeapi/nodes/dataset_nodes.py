from typing import Annotated, Optional

from nodeserver.api.node.node_parameters import DefaultDataTypes, Param
from nodeserver.api.node.nodes import NoInput, NoParameters
from nodeserver.api.node.slots import Input
from nodeserver.wrapper.metadata.nodes.node_metadata import NodeTypeMeta
from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror, SlotMirror
from pydantic import BaseModel
import torch

from torchserver.nodeapi.nodes.base_torch_nodes import _DatasetOut, BaseTorchNode, DatasetInputNode, TorchIOModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchserver.nodeapi.nodes.metadata import IMAGE_TAG, TORCH_DATASET_CATEGORY
from torchserver.nodeapi.nodes.torch_slots import TorchInput


class MNISTDatasetNode(DatasetInputNode):
    class Parameters(NoParameters): # type: ignore
        batch_size: Annotated[int, Param(
            label="Batch Size",
            step=1,
            datatype_override=DefaultDataTypes.UINT
        )] = 1

    _parameters: Parameters # type: ignore
    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_DATASET_CATEGORY,
        capitalized_name="",
        tags=[IMAGE_TAG]
    )
    
    dataloader: Optional[DataLoader]

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
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

    def forward(self, input: NoInput) -> _DatasetOut:
        path = self._mirror.data.get_parameter_value("path", "./data")
        batch_size = self._parameters.batch_size
        if path != self._last_path or batch_size != self._last_batch_size:
            self._load_dataset(path, batch_size)

        try:
            samples, labels = next(self.data_iter)
        except (StopIteration, TypeError):
            self.data_iter = iter(self.dataloader)
            samples, labels = next(self.data_iter)

        return _DatasetOut(
            sample=samples,
            target=torch.tensor(labels)
        )

class _SeededDataset(TorchIOModel):
    seed: Annotated[Optional[int], Input()]
    tensor_seed: Annotated[Optional[torch.Tensor], TorchInput()]

class SeededRandomDataset(DatasetInputNode):
    InputModel = _SeededDataset

    def forward(self, input: _SeededDataset) -> _DatasetOut: # type: ignore
        parsed_shape = tuple([int(dim_size) for dim_size in self._parameters.shape.split(",")])
        seed = input.seed
        if not input.seed:
            seed = input.tensor_seed
        
        torch.manual_seed(seed or 0)
        random_tensor = torch.randn(parsed_shape)
        return _DatasetOut(
            sample=random_tensor,
            target=random_tensor
        )

    _metadata: Optional[NodeTypeMeta] = NodeTypeMeta(
        category=TORCH_DATASET_CATEGORY,
        capitalized_name="",
        tags=[IMAGE_TAG]
    )