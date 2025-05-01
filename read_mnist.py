import struct
import numpy as np
import os

import random

from image_operations import ImageMutator

# --- Custom Exceptions ---
class MnistError(Exception):
  """Base exception for MNIST loading errors."""
  pass

class InvalidFileError(MnistError):
  """Error for invalid file format or structure."""
  pass

class IncompatibleDataError(MnistError):
  """Error for incompatible dimensions or counts between files."""
  pass

# --- MNIST Iterator Class ---
class MnistIterator:
  """
  Loads and iterates over the MNIST dataset from IDX files.

  Attributes:
    images (np.ndarray): A NumPy array containing the image data (count, rows, cols).
    labels (np.ndarray): A NumPy array containing the labels.
    rows (int): Number of rows per image.
    cols (int): Number of columns per image.
    count (int): Total number of images/labels in the dataset.
  """
  def __init__(self, images_path: str, labels_path: str):
    """
    Initializes the iterator by loading data from the specified paths.

    Args:
      images_path (str): Path to the IDX file containing image data.
      labels_path (str): Path to the IDX file containing label data.

    Raises:
      FileNotFoundError: If either file path does not exist.
      InvalidFileError: If file headers are invalid or file sizes are incorrect.
      IncompatibleDataError: If image and label counts or dimensions mismatch.
      struct.error: If there's an issue unpacking binary data.
    """
    # --- Load Images ---
    with open(images_path, 'rb') as f_img:
      # Read header (big-endian: magic, num_images, rows, cols)
      magic, num_images, self.rows, self.cols = struct.unpack(">IIII", f_img.read(16))
      if magic != 2051: raise InvalidFileError(f"Invalid magic number {magic} in images file '{images_path}'. Expected 2051.")

      expected_image_bytes = num_images * self.rows * self.cols
      image_data = f_img.read()
      if len(image_data) != expected_image_bytes: raise InvalidFileError(f"Image file '{images_path}' size mismatch. Expected {expected_image_bytes} data bytes, found {len(image_data)}.")
      self.images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, self.rows, self.cols)

    # --- Load Labels ---
    with open(labels_path, 'rb') as f_lbl:
      # Read header (big-endian: magic, num_labels)
      magic, num_labels = struct.unpack(">II", f_lbl.read(8))
      if magic != 2049: raise InvalidFileError(f"Invalid magic number {magic} in labels file '{labels_path}'. Expected 2049.")

      expected_label_bytes = num_labels
      label_data = f_lbl.read()
      if len(label_data) != expected_label_bytes: raise InvalidFileError(f"Label file '{labels_path}' size mismatch. Expected {expected_label_bytes} bytes, found {len(label_data)}.")

      self.labels = np.frombuffer(label_data, dtype=np.uint8)

    # --- Validation ---
    if num_images != num_labels: raise IncompatibleDataError(f"Number of images ({num_images}) and labels ({num_labels}) do not match.")

    self.count = num_images
    self._index = 0

  def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
    """
    Allows accessing items by index.

    Args:
      index (int): The index of the item to retrieve.

    Returns:
      tuple[np.ndarray, int]: The (image, label) tuple at the specified index.

    Raises:
      IndexError: If the index is out of bounds.
    """
    if 0 <= index < self.count:
      return (self.images[index], self.labels[index])
    else:
      raise IndexError(f"Index {index} is out of bounds for dataset with size {self.count}.")

  def __iter__(self):
    """
    Makes the object iterable. Resets the index and returns self.
    """
    self._index = 0
    return self

  def __next__(self) -> tuple[np.ndarray, int]:
    """
    Returns the next item in the iteration.

    Returns:
      tuple[np.ndarray, int]: The next (image, label) tuple.

    Raises:
      StopIteration: When the iteration is exhausted.
    """
    if self._index < self.count:
      item = self[self._index]
      self._index += 1
      return item
    else:
      raise StopIteration

  def __len__(self) -> int:
    """Returns the total number of items in the dataset."""
    return self.count

  def shuffle(self) -> None:
    """
    Shuffles the dataset (images and labels together) in place.
    Resets the iteration index.
    """
    # Create an array of indices from 0 to count-1
    indices = np.arange(self.count)
    # Shuffle the indices
    np.random.shuffle(indices)
    # Reorder images and labels using the shuffled indices
    self.images = self.images[indices]
    self.labels = self.labels[indices]

class RandomMnistIterator:
  """
  Iterates over a dataset (like MNIST) yielding a specified number of random samples.

  Handles its own random number generation internally.
  """
  def __init__(self, underlying, num_items: int):
    """
    Initializes the random iterator.

    Args:
      mnist_iterator (MnistIterator): The MnistIterator instance to draw samples from.
      num_items (int): The number of random items to iterate over.
    """
    self.underlying = underlying
    self.total_count = len(underlying)
    self.initial_count = num_items
    self.remaining = num_items

  def __iter__(self):
    """Makes the object iterable. Resets the remaining count."""
    self.remaining = self.initial_count
    return self

  def __next__(self) -> tuple[np.ndarray, int]:
    """
    Returns the next random item.

    Returns:
      tuple[np.ndarray, int]: A randomly selected (image, label) tuple.

    Raises:
      StopIteration: When the specified number of random items have been yielded.
    """
    if self.remaining == 0:
      raise StopIteration
    self.remaining -= 1
    return self.underlying[random.randint(0, self.total_count - 1)]

  def __len__(self) -> int:
    """Returns the total number of items in the dataset."""
    return self.initial_count

class EqualizedIterator:
  def __init__(self, underlying):
    self._underlying = underlying
    self.labels_counts = [0] * 10

  def __iter__(self):
    self.iterator = iter(self._underlying)
    self.underlying_done = False
    self.done = False
    self.labels_counts = [0] * 10
    return self

  def __next__(self) -> tuple[np.ndarray, int]:
    if self.done: raise StopIteration
    if self.underlying_done:
      while True:
        try:
          retval = next(self.iterator)
          if self.labels_counts[retval[1]] != 0:
            self.labels_counts[retval[1]] -= 1
            if max(self.labels_counts) == 0:
              self.done = True
            return retval
        except StopIteration:
          self.iterator = iter(self._underlying)
          continue

    try:
      value = next(self.iterator)
      self.labels_counts[value[1]] += 1
      return value
    except StopIteration:
      self.underlying_done = True
      minval = min(self.labels_counts)
      self.labels_counts = [l - minval for l in self.labels_counts]
      if max(self.labels_counts) == 0:
        self.done = True
        raise StopIteration
      return next(self)

  def __len__(self):
    return len(self.iterator)

class MutatorWrapped:
  def __init__(self, underlying, all_mutations: bool=True):
    self.underlying = underlying
    self.all_mutations = all_mutations

  def __iter__(self):
    self.iterator = iter(self.underlying)
    self.mutator = ImageMutator(self.underlying[0])
    self.mutator.randomlyMutate(self.all_mutations)

  def __next__(self):
    if self.mutator.images:
      return self.mutator.images.pop()
    else:
      self.mutator = ImageMutator(next(self.iterator))
      self.mutator.randomlyMutate(self.all_mutations)
      return self.mutator.images.pop()

class InvalidImageError(MnistError):
  """Error for invalid file format or structure."""
  pass

# --- Helper Function to Print Image ---
def print_image(image: np.ndarray) -> None:
  """
  Prints a 2D NumPy array (image) to the console using ASCII characters.

  Args:
    image (np.ndarray): A 2D NumPy array representing the image.
  """
  ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
  num_chars = len(ascii_chars)

  if image.size == 0:
    return
  if image.ndim != 2:
    raise InvalidImageError(f"Invalid number of dimensions for image, image must be 2d not {image.ndim}d")

  min_val = np.min(image)
  max_val = np.max(image)

  if max_val == min_val: # Handle flat image (all pixels the same)
    image = np.zeros(image.size)
  else:
    image = (image - min_val) / (max_val - min_val)

  char_indices = np.round(image * (num_chars - 1)).astype(int)

  for row_indices in char_indices:
    print("".join(ascii_chars[i] for i in row_indices))

dataset_dir = "dataset"
train_images_path = os.path.join(dataset_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(dataset_dir, "train-labels.idx1-ubyte")
mnist_train_iter = MnistIterator(train_images_path, train_labels_path)

test_images_path = os.path.join(dataset_dir, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(dataset_dir, "t10k-labels.idx1-ubyte")
mnist_test_iter = MnistIterator(test_images_path, test_labels_path)

# --- Example Usage / Test ---
if __name__ == "__main__":
  print("Attempting to load MNIST dataset...", end="")

  mnist_train = MnistIterator(train_images_path, train_labels_path)
  mnist_test = MnistIterator(test_images_path, test_labels_path)

  print("OK")

  print("First training image:")
  print_image(mnist_train[0][0])

  print("First testing image:")
  print_image(mnist_test[0][0])

