from typing import Optional

from nodeserver.wrapper.nodes.node.base_nodes import NodeMirror, SlotMirror, SlotOutput
from torchserver.nodeapi.nodes.base_torch_nodes import BaseTorchNode
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDatasetNode(BaseTorchNode):
    dataloader: Optional[DataLoader]

    def __init__(self, mirror: NodeMirror | None = None):
        super().__init__(mirror)
        self.bypass_cache = True
        
        self.dataset = None
        self.dataloader = None
        self.data_iter = None
        self._last_path = None
        self._last_batch_size = None

    def _load_dataset(self, path: str, batch_size: int):
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

    def forward(self, input: dict[SlotMirror, dict[SlotMirror, SlotOutput]]) -> dict[SlotMirror, SlotOutput]:
        path = self._mirror.data.get_parameter_value("path", "./data")
        batch_size = int(self._mirror.data.get_parameter_value("batch_size", 1))

        if path != self._last_path or batch_size != self._last_batch_size:
            self._load_dataset(path, batch_size)

        try:
            samples, labels = next(self.data_iter)
        except (StopIteration, TypeError):
            self.data_iter = iter(self.dataloader)
            samples, labels = next(self.data_iter)

        output_data = {
            "out_0": samples,
            "labels": labels
        }
        
        return self.map_to_slots(output_data)
    