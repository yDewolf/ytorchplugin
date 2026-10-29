import matplotlib.pyplot as plt
from nodeserver.engine.plugins.api.decorators.plugin_decorators import plugin_node
from nodeserver.engine.protocols.node.logic_nodes import BaseNode
from nodeserver.engine.runtime.context_injectors import ArtifactAwareInput
from nodeserver.engine.utils.context_managers import safe_atomic_save
import torch
from pathlib import Path

from core.datatypes import ArtifactDataType
from ytorchplugin.nodes.base_torch_node import TorchInputModel, TorchOutputModel

class PlotNodeInput(TorchInputModel, ArtifactAwareInput):
    tensor_data: torch.Tensor


class PlotNodeOutput(TorchOutputModel):
    plot_artifact: ArtifactDataType

@plugin_node(id="tensor_plot")
class MatplotlibPlotNode(BaseNode[PlotNodeInput, PlotNodeOutput]):
    InputModel = PlotNodeInput
    OutputModel = PlotNodeOutput

    def forward(self, inputs: PlotNodeInput) -> PlotNodeOutput:
        data_np = inputs.tensor_data.detach().cpu().numpy()
        node_uid = self.scene_data.uid

        filename, extension = "tensor_plot", "png"
        final_path: Path = inputs.scene_artifacts.get_node_artifact_path(
            node_uid=node_uid,
            filename=filename,
            hash=inputs.current_hash,
            extension=extension
        )

        try:
            with safe_atomic_save(final_path) as temp_path:
                fig, ax = plt.subplots(figsize=(6, 4))
                
                if data_np.ndim == 1:
                    ax.plot(data_np)
                elif data_np.ndim >= 2:
                    ax.imshow(data_np.squeeze(), cmap="viridis")
                else:
                    ax.plot([data_np.item()])
                    
                ax.set_title(f"Tensor Shape: {list(inputs.tensor_data.shape)}")
                ax.grid(True)
                
                fig.savefig(temp_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                
        except Exception as e:
            raise RuntimeError(f"Falha ao gerar o plot com Matplotlib: {e}")

        artifact_name = inputs.scene_artifacts.get_formatted_filename(filename, inputs.current_hash, extension)
        return PlotNodeOutput(
            plot_artifact=ArtifactDataType(
                artifact_name=artifact_name
            )
        )
