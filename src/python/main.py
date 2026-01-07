# import torch
import torch
import torch.nn as nn
import json
import os


from data_types.ModuleInstructions import InstructionHolder
from data_types.ModuleParts import ModulePart
from helpers.ShapeCalculator import ShapeCalculator
from helpers.ModelAssembler import ModelAssembler

model_data: dict = {}
with open(os.path.join(os.path.dirname(__file__), "..", "data", "model_format.json")) as file:
    model_data = json.load(file)

holder = InstructionHolder.from_dict(model_data)
print(holder.variable_controller.variables.keys())
print(holder.variable_controller.get_variable("modules:module_1:params:in_features"))

module_part = ModulePart(nn.Linear)
module_part.setup_module({"in_features": 20, "out_features": 30})
print(ShapeCalculator.calculate_out_shape(module_part))
print(ShapeCalculator.calculate_out_shape(module_part, (120, 20)))
print(ShapeCalculator.calculate_out_shape(module_part, (20, 120)))

conv2d_part = ModulePart(nn.Conv2d)
conv2d_part.setup_module({"in_channels": 120, "out_channels": 30, "kernel_size": (3, 3)})
print(ShapeCalculator.calculate_out_shape(conv2d_part))
print(ShapeCalculator.calculate_out_shape(conv2d_part, (120, 300, 300)))
print(ShapeCalculator.calculate_out_shape(conv2d_part, (30, 300, 300)))

model = ModelAssembler.assemble_model(holder)
pass
# for module in holder.modules:
#     print(holder.modules[module])