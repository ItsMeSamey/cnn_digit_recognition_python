from typing import Callable
import numpy as np
from abc import ABC, abstractmethod
from functions_activate import ActivationFunctionBase

class TestingLayerBase(ABC):
  @abstractmethod
  def reset(self, input_shape: tuple) -> tuple:
    """
    Randomly sets the weights and bisase for the layer, returns the output shape for the layer
    """
    pass

  @abstractmethod
  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Transforms the (non-batched) input, but does NOT cache anything
    """
    pass

  @abstractmethod
  def to_trainer(self) -> 'TrainingLayerBase':
    """
    Converts this to the Training variant of this layer.
    """
    pass

class TrainingLayerBase(ABC):
  @abstractmethod
  def reset_gradients(self):
    """
    Resets the gradients to zero.
    """
    pass

  @abstractmethod
  def reset(self, input_shape: tuple) -> tuple:
    """
    Randomly sets the weights and bisase for the layer, returns the output shape for the layer
    It calls reset on underlying tester as well as reset_gradients on itself.
    """
    pass

  @abstractmethod
  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Transforms (non-batched) input, caches the inputs for backward pass
    """
    pass

  @abstractmethod
  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    """
    Returns d_prev when calc_prev is set to true, otherwise returns None
    Stores the gradients (adds to them) internally, to be applied later by apply_gradient
    """
    pass

  @abstractmethod
  def apply_gradient(self, learning_rate):
    """
    Apply internally stored gradients
    """
    pass


  @abstractmethod
  def to_tester(self) -> 'TestingLayerBase':
    """
    Converts this to Testing variant of this layer
    """
    pass


class ConvolveTester(TestingLayerBase):
  def __init__(self, filter_x: int, filter_y: int, stride_x: int, stride_y: int, activation_function: ActivationFunctionBase) -> None:
    """
    Initializes the testing convolution layer parameters.

    Args:
      filter_x: Width of the convolutional filter.
      filter_y: Height of the convolutional filter.
      stride_x: Stride in the x-direction.
      stride_y: Stride in the y-direction.
      activation_function: The activation function to apply after convolution.
    """
    self.filter_x = filter_x
    self.filter_y = filter_y
    self.stride_x = stride_x
    self.stride_y = stride_y
    self.activation_function = activation_function

  def reset(self, input_shape: tuple) -> tuple:
    """
    Randomly initializes the filter and bias for the testing layer.

    Args:
      input_shape: The shape of the input data (height, width).

    Returns:
      The shape of the output data (output_height, output_width).
    """
    self.input_shape = input_shape
    input_h, input_w = input_shape
    filter_h, filter_w = self.filter_y, self.filter_x

    std_dev = np.sqrt(2.0 / (filter_h * filter_w)) # 'He' initialization scaling
    self.filter = np.random.randn(filter_h, filter_w) * std_dev
    self.bias = np.float64(0)

    # Calculate output dimensions
    self.out_h = input_h // self.stride_y
    self.out_w = input_w // self.stride_x
    if self.out_w * self.stride_x < input_w: self.out_w += 1
    if self.out_h * self.stride_y < input_h: self.out_h += 1

    self.output_shape = (self.out_h, self.out_w)
    return self.output_shape

  def convolve(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the convolution layer without caching.
    Does not apply activation function.

    Args:
      input_data: The input array of shape (height, width, channels).

    Returns:
      The output array after convolution and activation.
    """
    stride_y, stride_x = self.stride_y, self.stride_x

    output = np.zeros((self.out_h, self.out_w))
    filter_y_half = self.filter_y // 2
    filter_x_half = self.filter_x // 2
    input_data = np.pad(input_data, ((filter_y_half, filter_y_half), (filter_x_half, filter_x_half)), 'constant', constant_values=0)

    for out_y in range(self.out_h):
      for out_x in range(self.out_w):
        in_y_start = out_y * stride_y
        in_x_start = out_x * stride_x
        input_region = input_data[in_y_start : in_y_start + self.filter_y, in_x_start : in_x_start + self.filter_x]
        output[out_y, out_x] = np.sum(input_region * self.filter) + self.bias

    return output

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the convolution layer without caching.

    Args:
      input_data: The input array of shape (height, width, channels).

    Returns:
      The output array after convolution and activation.
    """
    return self.activation_function.forward(self.convolve(input_data))

  def to_trainer(self) -> TrainingLayerBase:
    """
    Converts this testing layer to a training layer.
    Requires filter and biass to have been initialized (e.g., by calling reset).
    """
    return ConvolveTrainer(self)

class ConvolveTrainer(TrainingLayerBase):
  def __init__(self, tester: ConvolveTester) -> None:
    """
    Initializes the training convolution layer.

    Args:
      tester: The corresponding ConvolveTester instance.
      input_shape: Optional initial input shape. filter and biass will be
             initialized when reset() is called with the actual input shape.
    """
    self.tester = tester
    self.filter_gradients: np.ndarray = np.zeros_like(self.tester.filter)
    self.bias_gradient: np.float64 = np.float64(0)

    self.cached_input: np.ndarray | None = None # Input to the layer
    self.cached_input_to_activation: np.ndarray | None = None # Output of convolution before activation
    self.cached_activated_output: np.ndarray | None = None # Output after activation

  def reset_gradients(self):
    """
    Resets all the gradients to zero.
    """
    self.filter_gradients = np.zeros_like(self.tester.filter)
    self.bias_gradient = np.float64(0)

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets the accumulated gradients, clears cached values, and
    initializes/re-initializes filter and biass based on the input shape.
    Calls reset on the underlying tester to handle weight/bias initialization.

    Args:
      input_shape: The shape of the input data (height, width, channels).

    Returns:
      The shape of the output data (output_height, output_width, num_filters).
    """
    retval = self.tester.reset(input_shape)
    self.reset_gradients()
    self.input_shape = input_shape

    self.cached_input = None
    self.cached_input_to_activation = None
    self.cached_activated_output = None

    return retval

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the convolution layer with caching.

    Args:
      input_data: The input array of shape (height, width, channels).

    Returns:
      The output array after convolution and activation.
    """
    self.cached_input = input_data
    output = self.tester.convolve(input_data)
    if (self.tester.activation_function.needs_cache_in()):
      self.cached_input_to_activation = output
    self.cached_activated_output = self.tester.activation_function.forward(output)
    return self.cached_activated_output

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    """
    Performs the backward pass for the convolution layer.
    Calculates gradients for filter and bias and accumulates them.
    Calculates and returns the gradient with respect to the input if calc_prev is True.

    Args:
      d_next: The gradient of the loss with respect to the output of this layer (after activation).
          Shape: (output_height, output_width).
    Returns:
      The gradient with respect to the input (d_prev) if calc_prev is True, otherwise None.
      Shape of d_prev: (input_height, input_width).
    """
    if self.cached_input is None or self.cached_activated_output is None or (self.tester.activation_function.needs_cache_in() and self.cached_input_to_activation is None):
      raise RuntimeError("Cached values are missing. Ensure forward() was called before backward().")
    if d_next.shape != (self.tester.out_h, self.tester.out_w):
      raise ValueError(f"Gradient d_next shape {d_next.shape} does not match expected output shape {(self.tester.out_h, self.tester.out_w)}.")

    d_output_convolve = self.tester.activation_function.backward(d_next, self.cached_input_to_activation, self.cached_activated_output)
    self.bias_gradient += np.sum(d_output_convolve)

    # if d_output_convolve.shape != (self.tester.out_h, self.tester.out_w):
    #    raise RuntimeError(f"Activation backward returned gradient with incorrect shape {d_output_convolve.shape}, expected {(self.tester.out_h, self.tester.out_w)}.")

    stride_y, stride_x = self.tester.stride_y, self.tester.stride_x
    input_data = self.cached_input
    input_h, input_w = input_data.shape

    filter_y_half = self.tester.filter_y // 2
    filter_x_half = self.tester.filter_x // 2

    d_prev = np.zeros_like(self.cached_input, dtype=d_output_convolve.dtype)

    for out_y in range(self.tester.out_h):
      for out_x in range(self.tester.out_w):
        sum_val = self.tester.bias

        for filter_y_offset in range(self.tester.filter_y):
          for filter_x_offset in range(self.tester.filter_x):
            effective_in_y = out_y * stride_y + filter_y_offset - filter_y_half
            effective_in_x = out_x * stride_x + filter_x_offset - filter_x_half

            if (0 <= effective_in_y < input_h and 0 <= effective_in_x < input_w):
              sum_val += d_output_convolve[out_y, out_x] * self.cached_input[effective_in_y, effective_in_x]

            if calc_prev:
              d_prev[effective_in_y, effective_in_x] += d_output_convolve[out_y, out_x] * self.tester.filter[filter_y_offset, filter_x_offset]

        self.filter_gradients[out_y, out_x] += sum_val

    self.cached_input = None
    self.cached_input_to_activation = None
    self.cached_activated_output = None
    if not calc_prev: return

    return d_prev

  def apply_gradient(self, learning_rate):
    """
    Apply internally stored gradients to update filter and bias.
    Resets gradients to zero after application.
    """
    # if self.filter_gradients is None or self.bias_gradient is None:
    #    raise RuntimeError("Gradients have not been calculated or are missing.")

    self.tester.filter -= learning_rate * self.filter_gradients
    self.tester.bias -= learning_rate * self.bias_gradient
    self.reset_gradients()

  def to_tester(self) -> TestingLayerBase:
    """
    Converts this training layer back to a testing layer.
    Returns the underlying ConvolveTester instance with the current filter and bias.
    """
    return self.tester

class DenseTester(TestingLayerBase):
  def __init__(self, out_width: int, activation_function: ActivationFunctionBase) -> None:
    """
    Initializes the testing dense layer parameters.

    Args:
      out_width: The number of neurons in the dense layer (output dimension).
      activation_function: The activation function to apply after the linear transformation.
    """
    self.out_width = out_width
    self.activation_function = activation_function

  def reset(self, input_shape: tuple) -> tuple:
    """
    Randomly initializes the weights and biases for the testing layer.

    Args:
      input_shape: The shape of the input data (width).

    Returns:
      The shape of the output data (out_width).
    """
    if len(input_shape) != 1: raise ValueError("Input shape must be a 1-dimensional tuple.")
    self.input_shape = input_shape
    self.input_width = input_shape[0]
    std_dev = np.sqrt(2.0 / self.input_width)
    self.weights = np.random.randn(self.out_width, self.input_width) * std_dev
    self.biases = np.zeros(self.out_width)
    self.output_shape = (self.out_width,)
    return self.output_shape

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the dense layer without caching.

    Args:
      input_data: The input array of shape matching self.input_shape.

    Returns:
      The output array after linear transformation and activation, shape (out_width,).
      Note: The TestingLayerBase forward is expected to return the shape calculated by reset,
      which is (1, out_width). Let's ensure the output matches this. The core calculation
      produces a (out_width,) shaped array, which can be reshaped to (1, out_width).
    """
    return self.activation_function.forward(np.dot(self.weights, input_data) + self.biases)

  def to_trainer(self) -> TrainingLayerBase:
    """
    Converts this testing layer to a training layer.
    Requires weights and biases to have been initialized (e.g., by calling reset).
    """
    return DenseTrainer(self)

class DenseTrainer(TrainingLayerBase):
  def __init__(self, tester: DenseTester) -> None:
    """
    Initializes the training dense layer.

    Args:
      tester: The corresponding DenseTester instance.
    """
    self.tester = tester

  def reset_gradients(self):
    """
    Resets all the gradients to zero.
    """
    self.weight_gradients = np.zeros_like(self.tester.weights)
    self.bias_gradients = np.zeros_like(self.tester.biases)

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets the accumulated gradients, clears cached values, and
    initializes/re-initializes weights and biases based on the input shape.
    Calls reset on the underlying tester to handle weight/bias initialization and output shape calculation.

    Args:
      input_shape: The shape of the input data.

    Returns:
      The shape of the output data (out_width).
    """
    retval = self.tester.reset(input_shape)
    self.reset_gradients()

    self.cached_input = None
    self.cached_input_to_activation = None
    self.cached_activated_output = None

    return retval

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass of the dense layer with caching.

    Args:
      input_data: The input array of shape matching self.input_shape.

    Returns:
      The output array after linear transformation and activation, shape (1, out_width).
    """
    if self.tester.weights is None or self.tester.biases is None or self.tester.input_shape is None:
      raise RuntimeError("Layer not properly initialized. Call reset() first.")
    if input_data.shape != self.tester.input_shape:
      raise ValueError(f"Input data shape {input_data.shape} does not match expected input shape {self.tester.input_shape}.")

    self.cached_input = input_data
    linear_output = np.dot(self.tester.weights, input_data) + self.tester.biases
    if self.tester.activation_function.needs_cache_in():
      self.cached_input_to_activation = linear_output

    self.cached_activated_output = self.tester.activation_function.forward(linear_output) # Shape (out_width,)
    return self.cached_activated_output

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    """
    Performs the backward pass for the dense layer.
    Calculates gradients for weights and bias and accumulates them.
    Calculates and returns the gradient with respect to the input if calc_prev is True.

    Args:
      d_next: The gradient of the loss with respect to the output of this layer (after activation).
          Expected shape is (out_width), matching the forward output shape.
          We will work with the flattened version (out_width,).
      calc_prev: If True, calculate and return the gradient with respect to the input.

    Returns:
      The gradient with respect to the input (d_prev) if calc_prev is True, otherwise None.
      Shape of d_prev will match the original input shape (self.input_shape).
    """
    if self.cached_input is None or self.cached_activated_output is None:
      raise RuntimeError("Cached values are missing. Ensure forward() was called before backward().")
    d_bias = self.tester.activation_function.backward(d_next, self.cached_input_to_activation, self.cached_activated_output)
    if d_bias.shape != (self.tester.out_width,):
      raise RuntimeError(f"Activation backward returned gradient with incorrect shape {d_bias.shape}, expected {(self.tester.out_width,)}.")

    d_weights = np.outer(d_bias, self.cached_input)
    # if d_weights.shape != (self.tester.out_width, self.tester.input_width):
    #   raise RuntimeError(f"Calculated weight gradient shape {d_weights.shape} does not match expected shape {(self.tester.out_width, self.tester.input_width)}.")

    self.weight_gradients += d_weights
    self.bias_gradients += d_bias

    self.cached_input = None
    self.cached_input_to_activation = None
    self.cached_activated_output = None
    if not calc_prev: return
    return np.dot(d_bias, self.tester.weights)

  def apply_gradient(self, learning_rate):
    """
    Apply internally stored gradients to update weights and biases.
    Resets gradients to zero after application.
    """
    self.tester.weights -= learning_rate * self.weight_gradients
    self.tester.biases -= learning_rate * self.bias_gradients

    # Reset accumulated gradients to zero
    self.weight_gradients = np.zeros_like(self.tester.weights)
    self.bias_gradients = np.zeros_like(self.tester.biases)

  def to_tester(self) -> TestingLayerBase:
    """
    Converts this training layer back to a testing layer.
    Returns the underlying DenseTester instance with the current weights and biases.
    """
    return self.tester


# --- Sequential Mering of layers ---

class SequentialTester(TestingLayerBase):
  """
  A composite layer that sequences multiple TestingLayerBase instances.
  Input is passed through layers sequentially.
  """
  def __init__(self, layers: list[TestingLayerBase]):
    """
    Initializes a sequential testing layer.

    Args:
      layers: A list of TestingLayerBase instances to be sequenced.
    """
    self.layers = layers

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets each layer in the sequence, chaining input/output shapes.

    Args:
      input_shape: The shape of the input to the first layer.

    Returns:
      The shape of the output from the last layer.
    """
    self.input_shape = input_shape
    for layer in self.layers: input_shape = layer.reset(input_shape)
    self.output_shape = input_shape
    return self.output_shape

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass through the sequence of layers.

    Args:
      input_data: The input data for the first layer.

    Returns:
      The output data from the last layer.
    """
    for layer in self.layers: input_data = layer.forward(input_data)
    return input_data

  def to_trainer(self) -> TrainingLayerBase:
    """
    Converts the sequential testing layer to a sequential training layer.
    """
    return SequentialTrainer(self)

class SequentialTrainer(TrainingLayerBase):
  """
  A composite layer that sequences multiple TrainingLayerBase instances.
  Input is passed through layers sequentially, and gradients flow backwards.
  """
  def __init__(self, tester: SequentialTester):
    """
    Initializes a sequential training layer.

    Args:
      layers: A list of TrainingLayerBase instances to be sequenced.
          Must contain at least one layer.
    """
    self.tester = tester
    self.layers = [layer.to_trainer() for layer in self.tester.layers]

  def reset_gradients(self):
    """
    Resets each layer in the sequence, chaining input/output shapes.
    """
    for layer in self.layers: layer.reset_gradients()

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets each layer in the sequence, chaining input/output shapes.
    Clears intermediate caches.

    Args:
      input_shape: The shape of the input to the first layer.

    Returns:
      The shape of the output from the last layer.
    """
    self.tester.input_shape = input_shape
    for layer in self.layers: input_shape = layer.reset(input_shape)
    self.tester.output_shape = input_shape
    return self.tester.output_shape

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass through the sequence of layers, caching intermediate values.

    Args:
      input_data: The input data for the first layer.

    Returns:
      The output data from the last layer.
    """
    for layer in self.layers: input_data = layer.forward(input_data)
    return input_data

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    """
    Performs the backward pass through the sequence of layers.

    Args:
      d_next: The gradient of the loss with respect to the output of this sequential layer.
      calc_prev: If True, calculate and return the gradient with respect to the input of this sequential layer.

    Returns:
      The gradient with respect to the input (d_prev) if calc_prev is True, otherwise None.
    """
    nullable_d_next: np.ndarray | None = d_next
    for i in reversed(range(len(self.layers))):
      layer = self.layers[i]
      if nullable_d_next is None: raise RuntimeError("layer.backward returned None, even with calc_prev=True.")
      nullable_d_next = layer.backward(nullable_d_next, calc_prev if i == 0 else True)
    return d_next

  def apply_gradient(self, learning_rate):
    """
    Applies accumulated gradients for all layers in the sequence.
    """
    for layer in self.layers: layer.apply_gradient(learning_rate)

  def to_tester(self) -> TestingLayerBase:
    """
    Converts the sequential training layer back to a sequential testing layer.
    """
    return self.tester

# --- merge_width (Parallel) ---

class ParallelTester(TestingLayerBase):
  """
  A composite layer that merges multiple TestingLayerBase instances in parallel.
  The same input is sent to all layers, and their outputs are concatenated.
  """
  def __init__(self, layers: list[TestingLayerBase]):
    """
    Initializes a parallel testing layer.

    Args:
      layers: A list of TestingLayerBase instances to be merged in parallel. Must contain at least one layer.
    """
    if not layers: raise ValueError("Parallel layer must contain at least one layer.")
    self.layers = layers

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets each parallel layer with the same input shape and calculates
    the combined output shape.

    Args:
      input_shape: The shape of the input to all parallel layers.

    Returns:
      The shape of the concatenated output from all layers.
    """
    self.input_shape = input_shape
    self.output_shape = sum([np.prod(layer.reset(input_shape)) for layer in self.layers])
    return (self.output_shape,)

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass through the parallel layers.

    Args:
      input_data: The input data for all parallel layers.

    Returns:
      The concatenated output data from all layers.
    """
    return np.concatenate([layer.forward(input_data).flatten() for layer in self.layers])

  def to_trainer(self) -> TrainingLayerBase:
    """
    Converts the parallel testing layer to a parallel training layer.
    """
    return ParallelTrainer(self)


class ParallelTrainer(TrainingLayerBase):
  """
  A composite layer that merges multiple TrainingLayerBase instances in parallel.
  The same input is sent to all layers, their outputs are concatenated,
  and gradients are split and summed.
  """
  def __init__(self, tester: ParallelTester):
    """
    Initializes a parallel training layer.

    Args:
      tester: The corresponding ParallelTester instance.
    """
    self.tester = tester
    self.layers = [layer.to_trainer() for layer in self.tester.layers]

  # Assuming individual trainer layers have a reset_gradients method
  def reset_gradients(self):
    """
    Resets accumulated gradients for all parallel layers.
    """
    for layer in self.layers: layer.reset_gradients()

  def reset(self, input_shape: tuple) -> tuple:
    """
    Resets each parallel layer with the same input shape and calculates
    the combined output shape. Resets accumulated gradients and clears caches.

    Args:
      input_shape: The shape of the input to all parallel layers.

    Returns:
      The shape of the concatenated output from all layers.
    """
    self.tester.input_shape = input_shape
    self.tester.output_shape = sum([np.prod(layer.reset(input_shape)) for layer in self.layers])
    return (self.tester.output_shape,)

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    """
    Performs the forward pass through the parallel layers, caching input.

    Args:
      input_data: The input data for all parallel layers.

    Returns:
      The concatenated output data from all layers.
    """
    return np.concatenate([layer.forward(input_data).flatten() for layer in self.layers])

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    """
    Performs the backward pass through the parallel layers.
    Sends the gradient and sends it to the corresponding layers.
    returns average of the gradients w.r.t. the input from each layer.

    Args:
      d_next: The gradient of the loss with respect to the concatenated output of this parallel layer.
      calc_prev: If True, calculate and return the gradient with respect to the input of this parallel layer.

    Returns:
      The gradient with respect to the input (d_prev) if calc_prev is True, otherwise None.
    """
    if not calc_prev:
      for layer in self.layers:
        offset = d_next.shape[0] - np.prod(layer.tester.output_shape)
        layer.backward(d_next[offset:].reshape(layer.tester.output_shape), False)
        d_next = d_next[:offset]
      return None
    else:
      retavl = np.zeros(self.tester.input_shape, dtype=d_next.dtype)
      for layer in self.layers:
        offset = d_next.shape[0] - np.prod(layer.tester.output_shape)
        retavl += layer.backward(d_next[offset:].reshape(layer.tester.output_shape), True) / len(self.layers)
        d_next = d_next[:offset]
      return retavl

  def apply_gradient(self, learning_rate):
    """
    Applies accumulated gradients for all parallel layers.
    """
    for layer in self.layers: layer.apply_gradient(learning_rate)

  def to_tester(self) -> TestingLayerBase:
    """
    Converts the parallel training layer back to a parallel testing layer.
    """
    return self.tester

class FlattenTester(TestingLayerBase):
  def __init__(self):
    pass

  def reset(self, input_shape: tuple) -> tuple:
    self.input_shape = input_shape
    self.output_shape = (int(np.prod(input_shape)),)
    return self.output_shape

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    return input_data.flatten()

  def to_trainer(self) -> TrainingLayerBase:
    return FlattenTrainer(self)

class FlattenTrainer(TrainingLayerBase):
  def __init__(self, tester: FlattenTester) -> None:
    self.tester = tester

  def reset_gradients(self):
    pass

  def reset(self, input_shape: tuple) -> tuple:
    return self.tester.reset(input_shape)

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    return self.tester.forward(input_data)

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    if not calc_prev: return None
    return d_next.reshape(self.tester.input_shape)

  def apply_gradient(self, learning_rate):
    pass

  def to_tester(self) -> TestingLayerBase:
    return self.tester

class LRFnWrappedTester(TestingLayerBase):
  def __init__(self, sublayer: TestingLayerBase, fn: Callable[[float, TestingLayerBase], float]) -> None:
    self.fn = fn
    self.sublayer = sublayer

  def reset(self, input_shape: tuple) -> tuple:
    return self.sublayer.reset(input_shape)

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    return self.sublayer.forward(input_data)

  def to_trainer(self) -> TrainingLayerBase:
    return LRFnWrappedTrainer(self)

class LRFnWrappedTrainer(TrainingLayerBase):
  def __init__(self, tester: LRFnWrappedTester) -> None:
    self.tester = tester
    self.sublayer = tester.sublayer.to_trainer()

  def reset_gradients(self):
    self.sublayer.reset_gradients()

  def reset(self, input_shape: tuple) -> tuple:
    return self.sublayer.reset(input_shape)

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    return self.sublayer.forward(input_data)

  def backward(self, d_next: np.ndarray, calc_prev: bool) -> np.ndarray | None:
    return self.sublayer.backward(d_next, calc_prev)

  def apply_gradient(self, learning_rate):
    self.sublayer.apply_gradient(self.tester.fn(learning_rate, self.tester.sublayer))

  def to_tester(self) -> TestingLayerBase:
    return self.tester

