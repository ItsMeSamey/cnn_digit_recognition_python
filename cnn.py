import numpy as np
from functions_loss import LossFunctionBase
from layers import TestingLayerBase, TrainingLayerBase

class CnnTester:
  def __init__(self, layer: TestingLayerBase, loss_gen: LossFunctionBase):
    self.layer = layer
    self.loss_gen = loss_gen
    pass

  def toTrainer(self) -> 'CnnTrainer':
    return CnnTrainer(self.layer.to_trainer(), self.loss_gen)

  def save(self):
    pass

  def exists(self):
    pass

  def load(self):
    pass

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

