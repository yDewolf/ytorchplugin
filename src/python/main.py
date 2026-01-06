# import torch
# import torch.nn as nn
import json
import os

from data_types.InstructionTypes import InstructionHolder

model_data: dict = {}
with open(os.path.join(os.path.dirname(__file__), "..", "data", "model_format.json")) as file:
    model_data = json.load(file)

holder = InstructionHolder.from_dict(model_data)
pass