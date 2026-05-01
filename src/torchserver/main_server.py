from nodeserver.api.base_server import NodeServer
from torchserver.nodeapi.torch_instance import TorchServerInstance

server = NodeServer(server_instance_type=TorchServerInstance)
server.run_server()
