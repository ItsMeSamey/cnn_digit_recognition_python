import struct
import numpy as np
import os

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

  def __len__(self) -> int:
    """Returns the total number of items in the dataset."""
    return self.count

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

  def has_next(self) -> bool:
    """Checks if there are more items to iterate over sequentially."""
    return self._index < self.count

  def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extracts the next sequential batch of images and labels.

    Advances the internal iterator index. If the remaining items are
    fewer than batch_size, it returns the remaining items. Returns None
    if the iterator has reached the end.

    Args:
      batch_size (int): The desired number of samples in the batch.

    Returns:
      Optional[List[Tuple[np.ndarray, int]]]: A list where each element
      is a tuple containing (image_array, label_int), or None if no
      more items are available.
    """

    if not self.has_next(): return None

    start_index = self._index
    end_index = min(start_index + batch_size, self.count)

    self._index = end_index
    return (self.images[start_index:end_index], self.labels[start_index:end_index])

  def random_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a random batch of images and labels sampled *with replacement*.

    This method does *not* affect the internal iterator index (_index).
    It returns separate arrays for images and labels for potential
    vectorized operations.

    Args:
      batch_size (int): The number of samples to include in the batch.

    Returns:
      tuple[np.ndarray, np.ndarray]: A tuple containing batch_images
      (batch_size, rows, cols) and batch_labels (batch_size,).
    """
    indices = np.random.choice(self.count, size=batch_size, replace=True)
    return (self.images[indices], self.labels[indices])

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


dataset_dir = "datasets"
train_images_path = os.path.join(dataset_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(dataset_dir, "train-labels.idx1-ubyte")
test_images_path = os.path.join(dataset_dir, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(dataset_dir, "t10k-labels.idx1-ubyte")

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

