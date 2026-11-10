from core.datatypes import ArtifactDataType

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from nodeblocks.engine.plugins.api.decorators import plugin_node
from nodeblocks.engine import BaseNode
from nodeblocks.engine.runtime import ArtifactAwareInput
from nodeblocks.engine.utils import safe_atomic_save
from pathlib import Path

from ytorchplugin.nodes.base_torch_node import TorchInputModel, TorchOutputModel

import torch

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
        final_path: Path = inputs.scene_artifacts.make_node_artifact_path(
            node_uid=node_uid,
            filename=filename,
            hash=inputs.current_hash,
            extension=extension
        )

        try:
            with safe_atomic_save(final_path) as temp_path:
                fig = Figure(figsize=(6, 4), dpi=50)
                ax = fig.add_subplot(111)
                
                if data_np.ndim == 1:
                    ax.plot(data_np, linewidth=1.0)
                elif data_np.ndim >= 2:
                    ax.imshow(data_np.squeeze(), cmap="viridis", aspect="auto")
                else:
                    ax.plot([data_np.item()])
                    
                ax.grid(True, linestyle="--", alpha=0.5)
                
                fig.savefig(temp_path, bbox_inches="tight", format="png")
                fig.clear()
                
        except Exception as e:
            raise RuntimeError(f"Falha ao gerar o plot com Matplotlib: {e}")

        inputs.scene_artifacts.commit_node_artifact(self.scene_data.uid, final_path)
        artifact_name = inputs.scene_artifacts.get_formatted_filename(filename, inputs.current_hash, extension)
        return PlotNodeOutput(
            plot_artifact=ArtifactDataType(
                artifact_name=artifact_name
            )
        )
