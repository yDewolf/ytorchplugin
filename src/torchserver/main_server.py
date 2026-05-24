from nodeserver.api.base_server import NodeServer
from torchserver.nodeapi.nodes.torch_types import TORCH_TYPES
from torchserver.nodeapi.torch_instance import TorchServerInstance
import torch
torch.set_default_dtype(torch.float32)

server = NodeServer(default_types=TORCH_TYPES, server_instance_type=TorchServerInstance)
server.run_server()
