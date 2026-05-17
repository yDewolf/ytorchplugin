from nodeserver.wrapper.nodes.data.node_metadata import NodeCategory, NodeTag

PYTORCH_TAG = NodeTag(tag_name="lib/PyTorch")
UTILITY_TAG = NodeTag(tag_name="PyTorch/utils")
DATASET_TAG = NodeTag(tag_name="PyTorch/dataset")
MODULE_TAG = NodeTag(tag_name="PyTorch/module")
GRADIENT_TAG = NodeTag(tag_name="PyTorch/gradient")
LOSS_TAG = NodeTag(tag_name="PyTorch/loss")
OPTIMIZER_TAG = NodeTag(tag_name="PyTorch/optimizer")

BASE_NODE_TAG = NodeTag(tag_name="PyTorch/base", description="Nodes that don't implement a specific behavior, they usually have more parameters")
IMAGE_TAG = NodeTag(tag_name="data/image", description="Nodes that use/outputs images (2D/3D Tensors)")

TENSOR_PROCESSOR = NodeTag(tag_name="Tensor/processor")

TORCH_NODE_CATEGORY = NodeCategory(
    name="TorchNodes",
    description="Nodes that implements PyTorch functions or modules",
    default_tags=[PYTORCH_TAG]
)

TORCH_UTIL_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    name="TorchUtil",
    description="Tensor utility functions from PyTorch",
    default_tags=[UTILITY_TAG]
)

TORCH_DATASET_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    name="TorchDataset",
    description="Nodes that pulls Samples and Targets from PyTorch datasets like MNIST",
    default_tags=[DATASET_TAG]
)

TORCH_MODULE_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    name="TorchModule",
    description="Nodes that implements PyTorch modules like Linear, Conv2d, etc",
    default_tags=[MODULE_TAG, TENSOR_PROCESSOR]
)

TORCH_GRADIENT_CATEGORY = NodeCategory(
    super_category=TORCH_NODE_CATEGORY,
    name="TorchGrad",
    description="Nodes that do backward / gradient update",
    default_tags=[GRADIENT_TAG]
)

TORCH_LOSS_CATEGORY = NodeCategory(
    super_category=TORCH_GRADIENT_CATEGORY,
    name="TorchLoss",
    description="Nodes that calculate how wrong the Model prediction is based on a Target",
    default_tags=[LOSS_TAG]
)

TORCH_TRAINER_CATEGORY = NodeCategory(
    super_category=TORCH_GRADIENT_CATEGORY,
    name="TorchTrainer",
    description="Nodes that receive Loss result and updates Modules weights and biases",
    default_tags=[OPTIMIZER_TAG]
)
