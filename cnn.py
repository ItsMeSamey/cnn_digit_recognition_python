import os
from sys import setswitchinterval
import numpy as np
from functions_loss import LossFunctionBase
from layers import ConvolveTester, DenseTester, FlattenTester, LRFnWrappedTester, ParallelTester, SequentialTester, TestingLayerBase, TrainingLayerBase
import json

def _parse_shape(info_str) -> tuple:
  if info_str.endswith("f64") or info_str.endswith("f32"):
    dims = info_str[:-3].strip("[]").split("][")
    return tuple(map(int, dims))
  raise ValueError(f"Unsupported format: {info_str}")

def _read_f64_array(fp, offset, shape) -> np.ndarray:
  fp.seek(offset)
  to_read = 8 * np.prod(shape)
  buffer = fp.read(8 * np.prod(shape))
  if len(buffer) != to_read: raise ValueError(f"Expected to read {to_read} bytes, got {len(buffer)} at offset {offset}")
  retval = np.frombuffer(buffer, dtype=np.float64).reshape(shape)
  return retval

# this is a recursive operartion done iteratively to avoid stack overflow
def _recursive_read_apply(fp, layer: TestingLayerBase, structure: dict):
  stack: list[tuple[TestingLayerBase, dict, int]] = [(layer, structure, 0)]
  while len(stack) != 0 :
    layer, structure, base_offset = stack.pop()
    name: str = structure["name"]
    offset: int = structure["offset"]
    info: str | list[dict] = structure["info"]

    if isinstance(layer, LRFnWrappedTester):
      stack.append((layer.sublayer, structure, base_offset))
      continue

    if (name == "layer" or name.isalnum()) and isinstance(info, list) and len(info) == 1:
      if len(info) != 1: raise ValueError(f"expected info length to be 1, got: {len(info)}")
      stack.append((layer, info[0], base_offset + offset))
      continue

    if name == "sequential_merged_layers" or name == "parallel_merged_layers":
      if name == "sequential_merged_layers": 
        if not isinstance(layer, SequentialTester): raise ValueError(f"expected SequentialTester, got {layer}")
      else:
        if not isinstance(layer, ParallelTester): raise ValueError(f"expected ParallelTester, got {layer}")

      if not isinstance(info, list): raise ValueError(f"expected a list, got {layer}")
      layers = []
      if name == "sequential_merged_layers":
        for l in layer.layers:
          if isinstance(l, FlattenTester): continue
          layers.append(l)
      else:
        layers = layer.layers
      if len(info) != len(layers): raise ValueError(f"length mismatch, json expects {len(info)} but layers have {len(layers)}")
      for i in reversed(range(len(info))):
        if int(info[i]["name"]) != i: raise ValueError(f"expected \"name\": {i} but got {info[i]}")
        stack.append((layers[i], info[i], base_offset + offset))
    elif name.isalnum() and isinstance(info, list) and len(info) == 2 and isinstance(info[0]["info"], str) and isinstance(info[1]["info"], str):
      if (info[0]["name"] == "filter" and info[1]["name"] == "bias") or (info[1]["name"] == "filter" and info[0]["name"] == "bias"):
        if info[1]["name"] == "filter" and info[0]["name"] == "bias": info[0], info[1] = info[1], info[0]
        if not isinstance(layer, ConvolveTester): raise ValueError(f"expected ConvolveTester layer, got {layer}")
        layer.filter = _read_f64_array(fp, base_offset + offset + info[0]["offset"], _parse_shape(info[0]["info"]))
        layer.bias = _read_f64_array(fp, base_offset + offset + info[1]["offset"], (1,))[0]
      elif (info[0]["name"] == "weights" and info[1]["name"] == "biases") or (info[1]["name"] == "weights" and info[0]["name"] == "biases"):
        if info[1]["name"] == "weights" and info[0]["name"] == "biases": info[0], info[1] = info[1], info[0]
        if not isinstance(layer, DenseTester): raise ValueError(f"expected ConvolveTester layer, got {layer}")
        layer.weights = _read_f64_array(fp, base_offset + offset + info[0]["offset"], _parse_shape(info[0]["info"]))
        layer.biases = _read_f64_array(fp, base_offset + offset + info[1]["offset"], _parse_shape(info[1]["info"]))
      else:
        raise ValueError(f"unknown value {structure} for {layer}")
    else:
      raise ValueError(f"unknown value {structure} for {layer}")

def _stringify_shape(shape: tuple) -> str:
  return "[" + "][".join(map(str, shape)) + "]" + "f64"

def _write_f64_array(fp, offset, array: np.ndarray):
  flat = array.flatten().astype('=f8')
  fp.seek(offset)
  fp.write(flat.tobytes())

# this could not be written iteratively with ease (not an issue for now)
def _recursive_write_apply(fd, layer: TestingLayerBase, structure: list, base_offset: int = 0) -> int:
  if isinstance(layer, FlattenTester):
    raise ValueError("unreachable: FlattenTester")
  elif isinstance(layer, SequentialTester) or isinstance(layer, ParallelTester):
    retval = 0
    layers = []
    for l in layer.layers:
      if isinstance(l, FlattenTester): continue
      layers.append(l)


    info = [{"name": str(i), "info": []} for i in range(len(layers))]
    structure.append({
      "name": "sequential_merged_layers" if isinstance(layer, SequentialTester) else "parallel_merged_layers",
      "offset": base_offset,
      "info": info,
    })

    for i in range(len(layers)):
      info[i]["offset"] = base_offset
      size = _recursive_write_apply(fd, layers[i], info[i]["info"], base_offset)
      base_offset += size
      retval += size
    return retval
  elif isinstance(layer, ConvolveTester):
    bias_offset = int(8 * np.prod(layer.filter.shape))
    structure.extend([
      {"name": "filter", "offset": 0, "info": _stringify_shape(layer.filter.shape)},
      {"name": "bias", "offset": bias_offset, "info": "f64"},
    ])
    _write_f64_array(fd, base_offset, layer.filter)
    _write_f64_array(fd, base_offset + bias_offset, np.array([layer.bias]))
    return int(bias_offset + 8)
  elif isinstance(layer, DenseTester):
    biases_offset = int(8 * np.prod(layer.weights.shape))
    structure.extend([
      {"name": "weights", "offset": 0, "info": _stringify_shape(layer.weights.shape)},
      {"name": "biases", "offset": biases_offset, "info": _stringify_shape(layer.biases.shape)},
    ])
    _write_f64_array(fd, base_offset, layer.weights)
    _write_f64_array(fd, base_offset + biases_offset, layer.biases)
    return int(biases_offset + 8 * np.prod(layer.biases.shape))
  elif isinstance(layer, LRFnWrappedTester):
    substructure = []
    structure.append({"name": "layer", "offset": 0, "info": substructure})
    return _recursive_write_apply(fd, layer.sublayer, substructure, base_offset)
  else:
    raise ValueError(f"unknown layer type {layer}")

class CnnTester:
  def __init__(self, input_shape: tuple, loss_gen: LossFunctionBase, layer: TestingLayerBase, hash: str):
    self.input_shape = input_shape
    self.loss_gen = loss_gen
    self.layer = layer
    self.layer.reset(input_shape)
    self.hash = hash

  def to_trainer(self) -> 'CnnTrainer':
    return CnnTrainer(self.input_shape, self.loss_gen, self.layer.to_trainer(), self.hash)

  def save(self):
    with open(f"model_{self.hash}.cnn", 'wb') as f:
      model_structure = []
      _recursive_write_apply(f, self.layer, model_structure)

    print(model_structure)
    with open(f"model_{self.hash}.json", 'w') as f:
      json.dump(model_structure, f, indent=2)

  def exists(self):
    return os.path.exists(f"model_{self.hash}.json") and os.path.exists(f"model_{self.hash}.cnn")

  def load(self):
    with open(f"model_{self.hash}.json", 'r') as f:
      model_structure = json.load(f)

    with open(f"model_{self.hash}.cnn", 'rb') as f:
      _recursive_read_apply(f, self.layer, model_structure[0])

  def predict(self, image: np.ndarray) -> np.ndarray:
    return self.layer.forward(image / 255.0)

  def test(self, iterator, verbose: bool = False) -> float:
    correct = 0
    incorrect = 0
    for image, label in iterator:
      result = self.predict(image)
      max_idx = np.argmax(result)
      if verbose: print(f"{label} -> {max_idx}: {result[max_idx]*100:.2f}")
      if label == max_idx:
        correct += 1
      else:
        incorrect += 1
    return correct / (correct + incorrect)

class CnnTrainer:
  def __init__(self, input_shape: tuple, loss_gen: LossFunctionBase, layer: TrainingLayerBase, hash: str):
    self.input_shape = input_shape
    self.loss_gen = loss_gen
    self.layer = layer
    self.layer.reset(input_shape)
    self.hash = hash

  def to_tester(self) -> 'CnnTester':
    return CnnTester(self.input_shape, self.loss_gen, self.layer.to_tester(), self.hash)

  def train(self, iterator, learning_rate: float, batch_size: int, verbose: bool = False):
    n = 0
    for image, label in iterator:
      predictions = self.layer.forward(image / 255.0)
      if verbose: print(f"({label} -> {np.argmax(predictions)}) = {self.loss_gen.forward(predictions, label)*100:.3f}")
      self.layer.backward(self.loss_gen.backward(predictions, label), False)
      n += 1
      if n == batch_size:
        self.layer.apply_gradient(learning_rate)
        n = 0
    if n > 0: self.layer.apply_gradient(learning_rate)

