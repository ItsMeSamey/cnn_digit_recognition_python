import os
import numpy as np
from functions_loss import LossFunctionBase
from layers import ConvolveTester, DenseTester, ParallelTester, SequentialTester, TestingLayerBase, TrainingLayerBase
import json

def parse_shape(info_str):
  if info_str.endswith("f64") or info_str.endswith("f32"):
    dims = info_str[:-3].strip("[]").split("][")
    return tuple(map(int, dims))
  raise ValueError(f"Unsupported format: {info_str}")

def read_f64_array(fp, offset, shape):
  size = np.prod(shape)
  fp.seek(offset)
  buffer = fp.read(8 * size)
  return np.frombuffer(buffer, dtype=np.float64).reshape(shape)

def write_f64_array(fp, offset, array: np.ndarray):
  flat = array.flatten().astype(np.float64)
  fp.seek(offset)
  fp.write(flat.tobytes())

def recursive_read_apply(fp, layer: TestingLayerBase, structure: dict):
  stack: list[tuple[TestingLayerBase, dict, int]] = [(layer, structure, 0)]
  while len(stack) != 0 :
    layer, structure, base_offset = stack.pop()
    name: str = structure["name"]
    offset: int = structure["offset"]
    info: str | list[dict] = structure["info"]

    if name == "layer":
      if not isinstance(info, list): raise ValueError(f"expected layer's info to be a list, got {info}")
      if len(info) != 1: raise ValueError(f"expected info length to be 1, got: {len(info)}")
      stack.append((layer, info[0], base_offset))
      continue
    elif name == "sequential_merged_layers" or name == "parallel_merged_layers":
      if name == "sequential_merged_layers": 
        if not isinstance(layer, SequentialTester): raise ValueError(f"expected SequentialTester, got {layer}")
      else:
        if not isinstance(layer, ParallelTester): raise ValueError(f"expected ParallelTester, got {layer}")

      if not isinstance(info, list): raise ValueError(f"expected a list, got {layer}")
      if len(info) != len(layer.layers): raise ValueError(f"length mismatch, json expects {len(info)} but layers have {len(layer.layers)}")
      for i in range(len(info)):
        if int(info[i]["name"]) != i: raise ValueError(f"extend \"name\": {i} as name but got {info[i]}")
        stack.append((layer.layers[i], info[i], base_offset + offset))
    elif name.isalnum() and isinstance(info, list) and len(info) == 2 and isinstance(info[0]["info"], str) and isinstance(info[1]["info"], str):

      if info[0]["name"] == "filter" and info[1] == "bias":
        if not isinstance(layer, ConvolveTester): raise ValueError(f"expected ConvolveTester layer, got {layer}")
        layer.filter = read_f64_array(fp, base_offset + offset + info[0]["offset"], parse_shape(info[0]["info"]))
        layer.bias = read_f64_array(fp, base_offset + offset + info[1]["offset"], (1,))[0]
      elif info[0]["name"] == "weights" and info[1] == "biases":
        if not isinstance(layer, DenseTester): raise ValueError(f"expected ConvolveTester layer, got {layer}")
        layer.weights = read_f64_array(fp, base_offset + offset + info[0]["offset"], parse_shape(info[0]["info"]))
        layer.biases = read_f64_array(fp, base_offset + offset + info[1]["offset"], parse_shape(info[1]["info"]))
    else:
      raise ValueError(f"unknown value {info}")

class CnnTester:
  def __init__(self, layer: TestingLayerBase, loss_gen: LossFunctionBase):
    self.layer = layer
    self.loss_gen = loss_gen
    pass

  def toTrainer(self) -> 'CnnTrainer':
    return CnnTrainer(self.layer.to_trainer(), self.loss_gen)

  def save(self, hash: str):
    return NotImplementedError("Not implemented")

  def exists(self, hash: str):
    return os.path.exists(f"model_{hash}.json") and os.path.exists(f"model_{hash}.cnn")

  def load(self, hash: str):
    with open(f"model_{hash}.json", 'r') as f:
      model_structure = json.load(f)

    with open(f"model_{hash}.cnn", 'r') as f:
      recursive_read_apply(f, self.layer, model_structure)

  def test(self, iterator) -> np.ndarray:
    correct = 0
    incorrect = 0
    for image, label in iterator:
      result = self.layer.forward(image)
      max_idx = np.argmax(result)
      if label == max_idx:
        correct += 1
      else:
        incorrect += 1
    return correct / (correct + incorrect)

class CnnTrainer:
  def __init__(self, layer: TrainingLayerBase, loss_gen: LossFunctionBase):
    self.layer = layer
    self.loss_gen = loss_gen

  def toTester(self) -> 'CnnTester':
    return CnnTester(self.layer.to_tester(), self.loss_gen)

  def train(self, iterator, learning_rate: int, batch_size: int):
    n = 0
    for image, label in iterator:
      predictions = self.layer.forward(image)
      self.layer.backward(self.loss_gen.backward(predictions, label), False)
      n += 1
      if n == batch_size:
        self.layer.apply_gradient(learning_rate)
        n = 0
    if n > 0: self.layer.apply_gradient(learning_rate)

