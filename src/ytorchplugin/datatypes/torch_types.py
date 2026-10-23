from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_datatype
from nodeserver.engine.plugins.api.plugin_datatype import PluginDatatype
from nodeserver.protocols.enums.datatype_enums import DefaultDataTypes, DefaultRenderers
import torch


@plugin_datatype(
    id="tensor",
    base_id=DefaultDataTypes.ARRAY,
    default_renderer=DefaultRenderers.ARRAY,
    aliases=[torch.Tensor]
)
class TensorDataType(PluginDatatype):
    def serialize(self):
        raise NotImplementedError()

    @classmethod
    def serialize_value(cls, value: torch.Tensor) -> list:
        # FIXME: talvez a gente precise modificar o serialize para incluir
        # o node que deu esse resultado
        with torch.no_grad():
            detached_tensor = value.detach().cpu()

            t_min, t_max = detached_tensor.min(), detached_tensor.max()
            if t_max > t_min:
                detached_tensor = (detached_tensor - t_min) / (t_max - t_min) * 255

            return detached_tensor.tolist()
