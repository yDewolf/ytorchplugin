from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_datatype
from nodeserver.engine.plugins.api.plugin_datatype import PluginDatatype
from nodeserver.protocols.enums.datatype_enums import DefaultDataTypes, DefaultRenderers
from nodeserver.protocols.manifest.node.datatype_value_meta import ArrayMetadata, SerializedDataTypeValue
import torch


@plugin_datatype(
    id="tensor",
    base_id=DefaultDataTypes.ARRAY,
    default_renderer=DefaultRenderers.ARRAY,
    aliases=[torch.Tensor]
)
class TensorDataType(PluginDatatype[ArrayMetadata]):
    def serialize(self):
        raise NotImplementedError()

    @classmethod
    def serialize_value(cls, value: torch.Tensor):
        # FIXME: talvez a gente precise modificar o serialize para incluir
        # o node que deu esse resultado
        with torch.no_grad():
            detached_tensor = value.detach().cpu()

            t_min, t_max = detached_tensor.min(), detached_tensor.max()
            if t_max > t_min:
                detached_tensor = (detached_tensor - t_min) / (t_max - t_min) * 255
            
            return SerializedDataTypeValue(
                value=detached_tensor.tolist(), 
                metadata=ArrayMetadata(
                    shape=detached_tensor.shape,
                    min=float(detached_tensor.min()),
                    max=float(detached_tensor.max())
                )
            )

@plugin_datatype(
    id="loss",
    base_id=DefaultDataTypes.ARRAY,
    default_renderer=DefaultRenderers.ARRAY,
)
class LossDataType(PluginDatatype[ArrayMetadata]):
    loss_tensor: torch.Tensor

    def __init__(self, loss: torch.Tensor) -> None:
        self.loss_tensor = loss

    def serialize(self):
        return self.serialize_value(self.loss_tensor)
    
    @classmethod
    def serialize_value(cls, value: torch.Tensor):
        with torch.no_grad():
            detached_tensor = value.detach().cpu()
            return SerializedDataTypeValue(
                value=detached_tensor.tolist(), metadata=ArrayMetadata(
                    shape=detached_tensor.shape,
                    min=float(detached_tensor.min()),
                    max=float(detached_tensor.max())
                )
            )
