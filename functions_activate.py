import numpy as np
from abc import ABC, abstractmethod

class ActivationFunctionBase(ABC):
  """
  Base class for all activation functions. Provides a common interface and initialization for derived classes.
  """
  @abstractmethod
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    """
    Forward pass of the activation function.
    """
    pass

  @abstractmethod
  def backward(self, derivative_array: np.ndarray, cache_in, cache_out: np.ndarray) -> np.ndarray:
    """
    Backward pass of the activation function.
    """
    pass

  @abstractmethod
  def needs_cache_in(self) -> bool:
    """
    The cache_in input will be None if this returns False
    """
    pass

class ReLU(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    return np.maximum(0, input_array)

  def backward(self, derivative_array: np.ndarray, cache_in: None, cache_out: np.ndarray) -> np.ndarray:
    return np.where(cache_out == 0, 0, derivative_array)

  def needs_cache_in(self) -> bool:
    return False

class PReLU(ActivationFunctionBase):
  def __init__(self, alpha: float):
    self.alpha = alpha

  def forward(self, input_array: np.ndarray) -> np.ndarray:
    return np.where(input_array < 0, input_array * self.alpha, input_array)

  def backward(self, derivative_array: np.ndarray, cache_in: np.ndarray | None, cache_out: np.ndarray) -> np.ndarray:
    if self.alpha < 0:
      if cache_in == None: raise TypeError("cache_in should not be null for negative alpha")
      return np.where(cache_in < 0, self.alpha * derivative_array, derivative_array)
    else:
      return np.where(cache_out < 0, self.alpha * derivative_array, derivative_array)

  def needs_cache_in(self) -> bool:
    return self.alpha < 0

class ELU(ActivationFunctionBase):
  def __init__(self, alpha: float):
    self.alpha = alpha

  def forward(self, input_array: np.ndarray) -> np.ndarray:
    return np.where(input_array < 0, self.alpha * (np.exp(input_array) - 1), input_array)

  def backward(self, derivative_array: np.ndarray, cache_in: np.ndarray, cache_out: np.ndarray) -> np.ndarray:
    return np.where(cache_in < 0, (cache_out + self.alpha) * derivative_array, derivative_array)

  def needs_cache_in(self) -> bool:
    return True

class Tanh(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    return np.tanh(input_array)

  def backward(self, derivative_array: np.ndarray, cache_in: None, cache_out: np.ndarray) -> np.ndarray:
    return (1 - cache_out * cache_out) * derivative_array

  def needs_cache_in(self) -> bool:
    return False

class Sigmoid(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-input_array))

  def backward(self, derivative_array: np.ndarray, cache_in: None, cache_out: np.ndarray) -> np.ndarray:
    return cache_out * (1 - cache_out) * derivative_array

  def needs_cache_in(self) -> bool:
    return False

class Softmax(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    exp_input = np.exp(input_array)
    return exp_input / np.sum(exp_input)

  def backward(self, derivative_array: np.ndarray, cache_in: None, cache_out: np.ndarray) -> np.ndarray:
    return cache_out * (1 - cache_out) * derivative_array

  def needs_cache_in(self) -> bool:
    return False

class Normalize(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    sum_input = np.sum(input_array)
    return input_array / sum_input

  def backward(self, derivative_array: np.ndarray, cache_in: np.ndarray, cache_out: np.ndarray) -> np.ndarray:
    return (cache_out / cache_in - cache_out) * derivative_array

  def needs_cache_in(self) -> bool:
    return True

class NormalizeAbsolute(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    sum_abs_input = np.sum(np.abs(input_array))
    return input_array / sum_abs_input

  def backward(self, derivative_array: np.ndarray, cache_in: np.ndarray, cache_out: np.ndarray) -> np.ndarray:
    return (cache_out / cache_in - np.abs(cache_out)) * derivative_array

  def needs_cache_in(self) -> bool:
    return True

class NormalizeSquared(ActivationFunctionBase):
  def forward(self, input_array: np.ndarray) -> np.ndarray:
    squared_input = input_array * input_array
    sum_squared_input = np.sum(squared_input)
    return squared_input / sum_squared_input

  def backward(self, derivative_array: np.ndarray, cache_in: np.ndarray, cache_out: np.ndarray) -> np.ndarray:
    return 2 * (cache_out / cache_in) * (1 - cache_out) * derivative_array

  def needs_cache_in(self) -> bool:
    return True
 
