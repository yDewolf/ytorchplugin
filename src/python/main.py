# import torch
import time
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
model = ModelAssembler.assemble_model(holder)
for input_key, value in model.forward_planner.get_required_inputs().items():
    print(input_key, value._input_shape)


FORWARD_COUNT = 1000
start = time.perf_counter()

# 91ms on cpu
# 213-160ms on cuda (probably because of cpu-gpu sync)
for i in range(FORWARD_COUNT):
    model.forward(
        {"something": torch.zeros(512, 42, device=model.device)}, 
        safe_mode=False
    )
    # print(i)

print(f"Performed {FORWARD_COUNT} forwards in {((time.perf_counter() - start) * 1000):.2f}ms")
pass
# for module in holder.modules:
#     print(holder.modules[module])