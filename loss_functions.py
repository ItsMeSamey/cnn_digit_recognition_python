import numpy as np
from abc import ABC, abstractmethod

class LossFunctionBase(ABC):
  """
  Base class for all loss functions.
  """
  @abstractmethod
  def forward(self, predictions: np.ndarray, target: int) -> float:
    pass

  @abstractmethod
  def backward(self, predictions: np.ndarray, target: int) -> np.ndarray:
    pass

class CategoricalCrossentropy(LossFunctionBase):
  def forward(self, predictions: np.ndarray, target: int) -> float:
    return -np.log(predictions[target])

  def backward(self, predictions: np.ndarray, target: int) -> np.ndarray:
    output = np.zeros_like(predictions)
    output[target] = -1 / predictions[target]
    return output

class MeanSquaredError(LossFunctionBase):
  def forward(self, predictions: np.ndarray, target: int) -> float:
    errors = np.where(np.arange(len(predictions)) == target, (predictions - 1)**2, predictions**2)
    return np.mean(errors)

  def backward(self, predictions: np.ndarray, target: int) -> np.ndarray:
    return np.where(np.arange(len(predictions)) == target, 2 * (predictions - 1), 2 * predictions)

class MeanAbsoluteError(LossFunctionBase):
  def forward(self, predictions: np.ndarray, target: int) -> float:
    errors = np.abs(np.where(np.arange(len(predictions)) == target, predictions - 1, predictions))
    return np.mean(errors)

  def backward(self, predictions: np.ndarray, target: int) -> np.ndarray:
    return np.where(predictions > np.where(np.arange(len(predictions)) == target, 1, 0), 1, -1)

