from nodeserver.api.instance.server_instance import ServerInstance

from torchserver.nodeapi.internal.torch_scene import TorchModelScene
from torchserver.nodeapi.nodes.torch_types import TORCH_TYPES
from torchserver.nodeapi.torch_server import TorchRuntime

class TorchServerInstance(ServerInstance): 
    def __init__(self, _default_types):
        # TODO Generate Types File
        types = TORCH_TYPES
        super().__init__(types)

        self._runtime = TorchRuntime()
        self._scene = TorchModelScene([], self.mirror_manager)
    