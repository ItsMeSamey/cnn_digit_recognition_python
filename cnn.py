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


class CnnTrainer:
  def __init__(self, layer: TrainingLayerBase, loss_gen: LossFunctionBase):
    self.layer = layer
    self.loss_gen = loss_gen

  def toTester(self) -> 'CnnTester':
    return CnnTester(self.layer.to_tester(), self.loss_gen)

