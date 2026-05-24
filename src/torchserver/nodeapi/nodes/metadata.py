
from nodeserver.wrapper.metadata.nodes.node_filters import NodeCategory, NodeTag


PYTORCH_TAG = NodeTag(tag_id="lib/PyTorch")
UTILITY_TAG = NodeTag(tag_id="PyTorch/utils")
DATASET_TAG = NodeTag(tag_id="PyTorch/dataset")
MODULE_TAG = NodeTag(tag_id="PyTorch/module")
GRADIENT_TAG = NodeTag(tag_id="PyTorch/gradient")
LOSS_TAG = NodeTag(tag_id="PyTorch/loss")
OPTIMIZER_TAG = NodeTag(tag_id="PyTorch/optimizer")

BASE_NODE_TAG = NodeTag(tag_id="PyTorch/base", description="Nodes that don't implement a specific behavior, they usually have more parameters")
IMAGE_TAG = NodeTag(tag_id="data/image", description="Nodes that use/outputs images (2D/3D Tensors)")

TENSOR_PROCESSOR = NodeTag(tag_id="Tensor/processor")

TORCH_NODE_CATEGORY = NodeCategory(
    category_id="Nodes",
    description="Nodes that implements PyTorch functions or modules",
    default_tags=[PYTORCH_TAG]
)

TORCH_UTIL_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    category_id="Util",
    description="Tensor utility functions from PyTorch",
    default_tags=[UTILITY_TAG]
)

TORCH_DATASET_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    category_id="Dataset",
    description="Nodes that pulls Samples and Targets from PyTorch datasets like MNIST",
    default_tags=[DATASET_TAG]
)

TORCH_MODULE_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    category_id="Module",
    description="Nodes that implements PyTorch modules like Linear, Conv2d, etc",
    default_tags=[MODULE_TAG, TENSOR_PROCESSOR]
)

TORCH_GRADIENT_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    category_id="Grad",
    description="Nodes that do backward / gradient update",
    default_tags=[GRADIENT_TAG]
)

TORCH_LOSS_CATEGORY = NodeCategory(
    super_category=TORCH_GRADIENT_CATEGORY,
    category_id="Loss",
    description="Nodes that calculate how wrong the Model prediction is based on a Target",
    default_tags=[LOSS_TAG]
)

TORCH_TRAINER_CATEGORY = NodeCategory(
    super_category=TORCH_GRADIENT_CATEGORY,
    category_id="Trainer",
    description="Nodes that receive Loss result and updates Modules weights and biases",
    default_tags=[OPTIMIZER_TAG]
)
