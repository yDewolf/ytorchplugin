# import torch
# import torch.nn as nn
import json
import os

from python.data_types.ModuleInstructions import InstructionHolder

model_data: dict = {}
with open(os.path.join(os.path.dirname(__file__), "..", "data", "model_format.json")) as file:
    model_data = json.load(file)

holder = InstructionHolder.from_dict(model_data)
print(holder.variable_controller.variables.keys())
print(holder.variable_controller.get_variable("modules:module_1:params:in_features"))

# for module in holder.modules:
#     print(holder.modules[module])