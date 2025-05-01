import numpy as np
import enum
import math
import random


class MutationTypes(enum.Enum):
  TranslateX = 0
  TranslateY = 1
  Scale = 2
  Rotate = 3
  Sharpen = 4
  Distort = 5
  Filter = 6
  Noise = 7
  Blur = 8
  BigSharpen = 9
  BigBlur = 10
  ScaleXY = 11

class EdgeBehaviour(enum.Enum):
  repeat = 0
  zero = 1
  max = 2


class ImageMutator:
  def __init__(self, image: np.ndarray):
    if image.ndim != 2 or image.dtype != np.uint8:
      raise ValueError("Input image must be a 2D NumPy array of dtype uint8")

    self.images = [image]
    self.ROWS, self.COLS = image.shape

  def randomlyMutate(self, all_mutations: bool) -> None:
    mutation_types = list(MutationTypes)
    selected_mutations = []

    if all_mutations:
      selected_mutations = mutation_types
    else:
      for mutation_type in mutation_types:
        if random.random() < 0.5:
           selected_mutations.append(mutation_type)

    random.shuffle(selected_mutations)

    for mutation_type in selected_mutations:
      self.applyMutation(mutation_type)


  def addImage(self, image: np.ndarray) -> None:
    if image.shape != (self.ROWS, self.COLS) or image.dtype != np.uint8:
       raise ValueError("Image to set must have the same shape and dtype as the original")
    self.images.append(image.copy())


  def applyMutation(self, mutation_type: MutationTypes) -> None:
    img_copy = self.images[-1].copy()
    mutated_image = None

    if mutation_type == MutationTypes.TranslateX:
      delta_y = random.randint(0, self.COLS // 4)
      direction = 'positive' if random.random() < 0.5 else 'negative'
      mutated_image = self.applyTranslateX(img_copy, delta_y, direction)
    elif mutation_type == MutationTypes.TranslateY:
      delta_x = random.randint(0, self.ROWS // 4)
      direction = 'positive' if random.random() < 0.5 else 'negative'
      mutated_image = self.applyTranslateY(img_copy, delta_x, direction)
    elif mutation_type == MutationTypes.Scale:
      scale_factor = random.random() * 0.8 - 0.5 + 1.0
      position_x = random.random()
      position_y = random.random()
      mutated_image = self.applyScale(img_copy, scale_factor, position_x, position_y)
    elif mutation_type == MutationTypes.Rotate:
      angle_radians = (random.random() - 0.5) * math.pi / 2.0
      pivot_x_percent = random.random() / 2.0 + 1.0 / 4.0
      pivot_y_percent = random.random() / 2.0 + 1.0 / 4.0
      mutated_image = self.applyRotate(img_copy, angle_radians, pivot_x_percent, pivot_y_percent)
    elif mutation_type == MutationTypes.Sharpen:
      sharpen_type = random.randint(0, 1)
      if sharpen_type == 0:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
      else:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
      self.automaticallyApplyConvolution(img_copy, kernel)
      return
    elif mutation_type == MutationTypes.Distort:
      strength = random.random() - 0.5
      if abs(strength) < 0.01:
        strength = math.copysign(0.01, strength) if strength != 0 else 0.01
      if strength > 0:
        strength *= 2.0
      mutated_image = self.applyBarrelPincushionDistort(img_copy, strength)
    elif mutation_type == MutationTypes.Filter:
      filter_type = random.randint(0, 2)
      if filter_type == 0:
        delta = random.randint(0, 256 // 4)
        direction = 'positive' if random.random() < 0.5 else 'negative'
        mutated_image = self.adjustBrightness(img_copy, delta, direction)
      elif filter_type == 1:
        pivot = random.random() * 255
        factor = random.random() * 2 + 0.1
        mutated_image = self.adjustContrast(img_copy, pivot, factor)
      else:
        gamma = random.random() * 1.8 + 0.1
        mutated_image = self.adjustGamma(img_copy, gamma)
    elif mutation_type == MutationTypes.Noise:
      noise_type = random.randint(0, 1)
      if noise_type == 0:
        std_dev = random.random() * 128
        mutated_image = self.addNoiseGaussian(img_copy, std_dev)
      else:
        probability = random.random() * 0.50
        mutated_image = self.addNoiseSaltAndPepper(img_copy, probability)
    elif mutation_type == MutationTypes.Blur:
      blur_type = random.randint(0, 3)
      if blur_type == 0:
        kernel_1d = np.array([1.0/4.0, 2.0/4.0, 1.0/4.0], dtype=np.float32)
        self.applySapetableConvolution(img_copy, kernel_1d, kernel_1d)
        return
      elif blur_type == 1:
        kernel_1d = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
        self.applySapetableConvolution(img_copy, kernel_1d, kernel_1d)
        return
      elif blur_type == 2:
        kernel_1d = np.array([1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0], dtype=np.float32)
        direction = random.randint(0, 1)
        edge_behaviour = random.choice(list(EdgeBehaviour))
        if direction == 0:
          mutated_image = self.applyRowConvolution(img_copy, kernel_1d, edge_behaviour)
        else:
          mutated_image = self.applyColumnConvolution(img_copy, kernel_1d, edge_behaviour)
      elif blur_type == 3:
        kernel = np.array([
          [0, 1.0/12.0, 1.0/12.0, 0],
          [1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0],
          [1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0],
          [0, 1.0/12.0, 1.0/12.0, 0]
        ], dtype=np.float32)
        self.automaticallyApplyConvolution(img_copy, kernel)
        return
    elif mutation_type == MutationTypes.BigSharpen:
      sharpen_type = random.randint(0, 3)
      if sharpen_type == 0:
        kernel = np.array([
          [0, 0, -1, 0, 0],
          [0, 0, -1, 0, 0],
          [-1, -1, 9, -1, -1],
          [0, 0, -1, 0, 0],
          [0, 0, -1, 0, 0]
        ], dtype=np.float32)
      elif sharpen_type == 1:
        kernel = np.array([
          [-1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1],
          [-1, -1, 25, -1, -1],
          [-1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1]
        ], dtype=np.float32)
      elif sharpen_type == 2:
        kernel = np.array([
          [0, 0, -1, 0, 0],
          [0, -1, -1, -1, 0],
          [-1, -1, 9, -1, -1],
          [0, -1, -1, -1, 0],
          [0, 0, -1, 0, 0]
        ], dtype=np.float32)
      else:
        kernel = np.array([
          [0, -1, -1, -1, 0],
          [-1, -1, -1, -1, -1],
          [-1, -1, 21, -1, -1],
          [-1, -1, -1, -1, -1],
          [0, -1, -1, -1, 0]
        ], dtype=np.float32)
      self.automaticallyApplyConvolution(img_copy, kernel)
      return
    elif mutation_type == MutationTypes.BigBlur:
      blur_type = random.randint(0, 1)
      if blur_type == 0:
        kernel_1d = np.array([1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0], dtype=np.float32)
        self.applySapetableConvolution(img_copy, kernel_1d, kernel_1d)
        return
      else:
        kernel_1d = np.array([1.0/5.0, 1.0/5.0, 1.0/5.0, 1.0/5.0, 1.0/5.0], dtype=np.float32)
        self.applySapetableConvolution(img_copy, kernel_1d, kernel_1d)
        return
    elif mutation_type == MutationTypes.ScaleXY:
      scale_factor_x = random.random() * 0.6 - 0.35 + 1.0
      scale_factor_y = random.random() * 0.6 - 0.35 + 1.0
      position_x = random.random()
      position_y = random.random()
      mutated_image = self.applyScaleXY(img_copy, scale_factor_x, scale_factor_y, position_x, position_y)
    else:
      raise ValueError(f"Unhandled mutation type: {mutation_type}")

    if mutated_image is None:
      raise ValueError(f"Mutation type {mutation_type} returned None")

    self.addImage(mutated_image)


  def applyTranslateX(self, image: np.ndarray, delta_y: int, direction: str) -> np.ndarray:
    output_image = np.zeros_like(image, dtype=np.uint8)
    if direction == 'positive':
      output_image[:, delta_y:] = image[:, :self.COLS - delta_y]
    else:
      output_image[:, :self.COLS - delta_y] = image[:, delta_y:]
    return output_image


  def applyTranslateY(self, image: np.ndarray, delta_x: int, direction: str) -> np.ndarray:
    output_image = np.zeros_like(image, dtype=np.uint8)
    if direction == 'positive':
      output_image[delta_x:, :] = image[:self.ROWS - delta_x, :]
    else:
      output_image[:self.ROWS - delta_x, :] = image[delta_x:, :]
    return output_image


  def applyScale(self, image: np.ndarray, scale_factor: float, position_x: float, position_y: float) -> np.ndarray:
    return self.applyScaleXY(image, scale_factor, scale_factor, position_x, position_y)


  def applyScaleXY(self, image: np.ndarray, scale_factor_x: float, scale_factor_y: float, position_x: float, position_y: float) -> np.ndarray:
    assert 0 <= position_x <= 1
    assert 0 <= position_y <= 1

    output_image = np.zeros_like(image, dtype=np.uint8)

    scaled_rows = self.ROWS * scale_factor_y
    scaled_cols = self.COLS * scale_factor_x

    max_crop_top = scaled_rows - self.ROWS
    max_crop_left = scaled_cols - self.COLS

    crop_top_scaled = position_y * max_crop_top
    crop_left_scaled = position_x * max_crop_left

    r_out, c_out = np.indices((self.ROWS, self.COLS))

    r_scaled = r_out.astype(np.float32) + crop_top_scaled
    c_scaled = c_out.astype(np.float32) + crop_left_scaled

    r_in = r_scaled / scale_factor_y
    c_in = c_scaled / scale_factor_x

    valid_mask = (r_in >= 0) & (r_in < self.ROWS) & (c_in >= 0) & (c_in < self.COLS)

    output_image[valid_mask] = self.interpolateBilinear(image, r_in[valid_mask], c_in[valid_mask])

    return output_image


  def applyRotate(self, image: np.ndarray, angle_radians: float, pivot_x_percent: float, pivot_y_percent: float) -> np.ndarray:
    output_image = np.zeros_like(image, dtype=np.uint8)

    pivot_x = (self.COLS - 1) * pivot_x_percent
    pivot_y = (self.ROWS - 1) * pivot_y_percent

    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)

    r_out, c_out = np.indices((self.ROWS, self.COLS))

    r_out_translated = r_out.astype(np.float32) - pivot_y
    c_out_translated = c_out.astype(np.float32) - pivot_x

    r_in = r_out_translated * cos_a - c_out_translated * sin_a + pivot_y
    c_in = r_out_translated * sin_a + c_out_translated * cos_a + pivot_x

    valid_mask = (r_in >= 0) & (r_in < self.ROWS) & (c_in >= 0) & (c_in < self.COLS)

    output_image[valid_mask] = self.interpolateBilinear(image, r_in[valid_mask], c_in[valid_mask])

    return output_image


  def applyBarrelPincushionDistort(self, image: np.ndarray, strength: float) -> np.ndarray:
    if strength == 0:
      return image.copy()

    output_image = np.zeros_like(image, dtype=np.uint8)

    center_x = self.COLS / 2.0
    center_y = self.ROWS / 2.0
    max_radius = math.sqrt(center_x * center_x + center_y * center_y)

    r_out, c_out = np.indices((self.ROWS, self.COLS))

    x_out = c_out.astype(np.float32) - center_x
    y_out = r_out.astype(np.float32) - center_y

    dist = np.sqrt(x_out * x_out + y_out * y_out)

    distortion_factor = strength * np.power(dist / max_radius, 2.0) if max_radius > 0 else 0

    r_in_distorted = dist * (1.0 + distortion_factor)

    angle = np.arctan2(y_out, x_out)
    angle[dist == 0] = 0

    x_in = r_in_distorted * np.cos(angle)
    y_in = r_in_distorted * np.sin(angle)

    c_in = x_in + center_x
    r_in = y_in + center_y

    valid_mask = (r_in >= 0) & (r_in < self.ROWS) & (c_in >= 0) & (c_in < self.COLS)

    output_image[valid_mask] = self.interpolateBilinear(image, r_in[valid_mask], c_in[valid_mask])

    return output_image


  def interpolateBilinear(self, image: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    clamped_r = np.clip(r, 0, self.ROWS - 1)
    clamped_c = np.clip(c, 0, self.COLS - 1)

    r_base = np.floor(clamped_r).astype(int)
    c_base = np.floor(clamped_c).astype(int)

    dr = clamped_r - r_base
    dc = clamped_c - c_base

    R1 = r_base
    R2 = np.minimum(self.ROWS - 1, r_base + 1)
    C1 = c_base
    C2 = np.minimum(self.COLS - 1, c_base + 1)

    img_R1_C1 = image[R1, C1].astype(np.float32)
    img_R1_C2 = image[R1, C2].astype(np.float32)
    img_R2_C1 = image[R2, C1].astype(np.float32)
    img_R2_C2 = image[R2, C2].astype(np.float32)

    interpolated_top = (1.0 - dc) * img_R1_C1 + dc * img_R1_C2
    interpolated_bottom = (1.0 - dc) * img_R2_C1 + dc * img_R2_C2

    interpolated_value = (1.0 - dr) * interpolated_top + dr * interpolated_bottom

    return np.clip(np.round(interpolated_value), 0, 255).astype(np.uint8)


  def adjustBrightness(self, image: np.ndarray, delta: int, direction: str) -> np.ndarray:
    if direction == 'positive':
      return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    else:
      return np.clip(image.astype(np.int16) - delta, 0, 255).astype(np.uint8)


  def adjustContrast(self, image: np.ndarray, pivot: float, factor: float) -> np.ndarray:
    img_float = image.astype(np.float32)

    adjusted_image = (img_float - pivot) * factor + pivot

    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


  def adjustGamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
      raise ValueError("Gamma value must be positive")

    gamma_lookup_table = np.power(np.arange(256) / 255.0, gamma) * 255.0
    gamma_lookup_table = np.clip(gamma_lookup_table, 0, 255).astype(np.uint8)

    return gamma_lookup_table[image]


  def addNoiseGaussian(self, image: np.ndarray, std_dev: float) -> np.ndarray:
    noised_image = image.astype(np.float32) + np.random.normal(0, std_dev, image.shape)
    return np.clip(noised_image, 0, 255).astype(np.uint8)


  def addNoiseSaltAndPepper(self, image: np.ndarray, probability: float) -> np.ndarray:
    output_image = image.copy()
    random_mask = np.random.rand(self.ROWS, self.COLS)
    output_image[random_mask < probability / 2] = 0
    output_image[(random_mask >= probability / 2) & (random_mask < probability)] = 255
    return output_image


  def applyConvolution(self, image: np.ndarray, kernel: np.ndarray, edge_behaviour: EdgeBehaviour) -> np.ndarray:
    kernel_rows, kernel_cols = kernel.shape
    row_offset = kernel_rows // 2
    col_offset = kernel_cols // 2

    output_image = np.zeros_like(image, dtype=np.float32)

    padded_image = None
    if edge_behaviour == EdgeBehaviour.repeat:
      padded_image = np.pad(image, ((row_offset, row_offset), (col_offset, col_offset)), mode='edge')
    elif edge_behaviour == EdgeBehaviour.zero:
      padded_image = np.pad(image, ((row_offset, row_offset), (col_offset, col_offset)), mode='constant', constant_values=0)
    elif edge_behaviour == EdgeBehaviour.max:
       padded_image = np.pad(image, ((row_offset, row_offset), (col_offset, col_offset)), mode='constant', constant_values=255)

    for r in range(self.ROWS):
      for c in range(self.COLS):
        roi = padded_image[r:r + kernel_rows, c:c + kernel_cols]

        output_image[r, c] = np.sum(roi.astype(np.float32) * kernel)

    return np.clip(output_image, 0, 255).astype(np.uint8)


  def automaticallyApplyConvolution(self, image: np.ndarray, kernel: np.ndarray) -> None:
    edge_behaviour = random.choice(list(EdgeBehaviour))
    mutated_image = self.applyConvolution(image, kernel, edge_behaviour)
    self.addImage(mutated_image)


  def applyRowConvolution(self, image: np.ndarray, kernel_r: np.ndarray, edge_behaviour: EdgeBehaviour) -> np.ndarray:
    kernel_len = kernel_r.size
    col_offset = kernel_len // 2

    output_image = np.zeros_like(image, dtype=np.float32)

    padded_image = None
    if edge_behaviour == EdgeBehaviour.repeat:
      padded_image = np.pad(image, ((0, 0), (col_offset, col_offset)), mode='edge')
    elif edge_behaviour == EdgeBehaviour.zero:
      padded_image = np.pad(image, ((0, 0), (col_offset, col_offset)), mode='constant', constant_values=0)
    elif edge_behaviour == EdgeBehaviour.max:
       padded_image = np.pad(image, ((0, 0), (col_offset, col_offset)), mode='constant', constant_values=255)

    for r in range(self.ROWS):
      for c in range(self.COLS):
        roi = padded_image[r, c:c + kernel_len]

        output_image[r, c] = np.sum(roi.astype(np.float32) * kernel_r)

    return np.clip(output_image, 0, 255).astype(np.uint8)


  def applyColumnConvolution(self, image: np.ndarray, kernel_c: np.ndarray, edge_behaviour: EdgeBehaviour) -> np.ndarray:
    kernel_len = kernel_c.size
    row_offset = kernel_len // 2

    output_image = np.zeros_like(image, dtype=np.float32)

    padded_image = None
    if edge_behaviour == EdgeBehaviour.repeat:
      padded_image = np.pad(image, ((row_offset, row_offset), (0, 0)), mode='edge')
    elif edge_behaviour == EdgeBehaviour.zero:
      padded_image = np.pad(image, ((row_offset, row_offset), (0, 0)), mode='constant', constant_values=0)
    elif edge_behaviour == EdgeBehaviour.max:
       padded_image = np.pad(image, ((row_offset, row_offset), (0, 0)), mode='constant', constant_values=255)

    for r in range(self.ROWS):
      for c in range(self.COLS):
        roi = padded_image[r:r + kernel_len, c]

        output_image[r, c] = np.sum(roi.astype(np.float32) * kernel_c)

    return np.clip(output_image, 0, 255).astype(np.uint8)


  def applySapetableConvolution(self, image: np.ndarray, kernel_r: np.ndarray, kernel_c: np.ndarray) -> None:
    edge_behaviour_r = random.choice(list(EdgeBehaviour))
    edge_behaviour_c = random.choice(list(EdgeBehaviour))

    if random.random() < 0.5:
      intermediate_image = self.applyColumnConvolution(image, kernel_c, edge_behaviour_c)
      final_image = self.applyRowConvolution(intermediate_image, kernel_r, edge_behaviour_r)
    else:
      intermediate_image = self.applyRowConvolution(image, kernel_r, edge_behaviour_r)
      final_image = self.applyColumnConvolution(intermediate_image, kernel_c, edge_behaviour_c)

    self.addImage(final_image)


if __name__ == '__main__':
  import unittest
  from read_mnist import MnistIterator

  def create_test_image(rows: int, cols: int, value: int) -> np.ndarray:
    return np.full((rows, cols), value, dtype=np.uint8)

  def create_gradient_image(rows: int, cols: int) -> np.ndarray:
    image = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
      for c in range(cols):
        image[r, c] = int(((r / (rows - 1)) + (c / (cols - 1))) / 2.0 * 255.0)
    return image

  class TestImageMutator(unittest.TestCase):

    def test_instantiation_and_initial_state(self):
      rows, cols = 10, 20
      dummy_image = create_test_image(rows, cols, 100)
      mutator = ImageMutator(dummy_image)

      self.assertEqual(len(mutator.images), 1)
      self.assertEqual(len(mutator.images), 1)
      self.assertEqual(mutator.images[0].shape, (rows, cols))
      self.assertEqual(mutator.images[0].dtype, np.uint8)
      self.assertEqual(mutator.ROWS, rows)
      self.assertEqual(mutator.COLS, cols)

    def test_randomlyMutate_runs_and_increments_image_count(self):
      rows, cols = 10, 10
      dummy_image = create_test_image(rows, cols, 128)
      mutator = ImageMutator(dummy_image)

      random.seed(0)
      np.random.seed(0)

      initial_image_count = len(mutator.images)
      mutator.randomlyMutate(all_mutations=True)

      self.assertGreater(len(mutator.images), initial_image_count)

      for i in range(initial_image_count, len(mutator.images)):
         self.assertEqual(mutator.images[i].shape, (rows, cols))
         self.assertEqual(mutator.images[i].dtype, np.uint8)


    def test_applyTranslateX_shifts_image_right(self):
      rows, cols = 3, 3
      image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], dtype=np.uint8)
      expected = np.array([
        [0, 1, 2],
        [0, 4, 5],
        [0, 7, 8],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      translated_image = mutator.applyTranslateX(image, 1, 'positive')
      np.testing.assert_array_equal(expected, translated_image)

    def test_applyTranslateX_shifts_image_left(self):
      rows, cols = 3, 3
      image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], dtype=np.uint8)
      expected = np.array([
        [2, 3, 0],
        [5, 6, 0],
        [8, 9, 0],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      translated_image = mutator.applyTranslateX(image, 1, 'negative')
      np.testing.assert_array_equal(expected, translated_image)

    def test_applyTranslateY_shifts_image_down(self):
      rows, cols = 3, 3
      image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], dtype=np.uint8)
      expected = np.array([
        [0, 0, 0],
        [1, 2, 3],
        [4, 5, 6],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      translated_image = mutator.applyTranslateY(image, 1, 'positive')
      np.testing.assert_array_equal(expected, translated_image)

    def test_applyTranslateY_shifts_image_up(self):
      rows, cols = 3, 3
      image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], dtype=np.uint8)
      expected = np.array([
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      translated_image = mutator.applyTranslateY(image, 1, 'negative')
      np.testing.assert_array_equal(expected, translated_image)

    def test_interpolateBilinear_handles_integer_coordinates(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)

      self.assertEqual(mutator.interpolateBilinear(image, np.array([0.0]), np.array([0.0]))[0], 10)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([1.0]), np.array([1.0]))[0], 50)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([2.0]), np.array([2.0]))[0], 90)

    def test_interpolateBilinear_handles_fractional_coordinates(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 20],
        [30, 40],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)

      self.assertEqual(mutator.interpolateBilinear(image, np.array([0.5]), np.array([0.5]))[0], 25)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([0.25]), np.array([0.75]))[0], 22)

    def test_interpolateBilinear_handles_boundary_clamping(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 20],
        [30, 40],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)

      self.assertEqual(mutator.interpolateBilinear(image, np.array([-0.1]), np.array([-0.1]))[0], 10)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([-0.1]), np.array([1.1]))[0], 20)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([1.1]), np.array([-0.1]))[0], 30)
      self.assertEqual(mutator.interpolateBilinear(image, np.array([1.1]), np.array([1.1]))[0], 40)

    def test_applyScale_scales_down(self):
      rows, cols = 4, 4
      image = create_gradient_image(rows, cols)

      expected_approx = np.array([
        [0, 85, 0, 0],
        [85, 170, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      scaled_image = mutator.applyScale(image, 0.5, 0.0, 0.0)

      np.testing.assert_array_equal(expected_approx, scaled_image)

    def test_applyScale_scales_up(self):
      rows, cols = 2, 2
      image = np.array([
        [0, 100],
        [200, 255],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      scaled_image = mutator.applyScale(image, 2.0, 0.5, 0.5)

      self.assertEqual(scaled_image.shape, (rows, cols))
      self.assertEqual(scaled_image.dtype, np.uint8)

      scaled_image_tl = mutator.applyScale(image, 2.0, 0.0, 0.0)

      self.assertFalse(np.array_equal(image, scaled_image_tl))
      self.assertTrue(np.any(scaled_image_tl == 0))

    def test_applyRotate_rotates_by_0_degrees(self):
      rows, cols = 3, 3
      image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      rotated_image = mutator.applyRotate(image, 0.0, 0.5, 0.5)

      np.testing.assert_array_almost_equal(image, rotated_image)

    def test_applyRotate_rotates_by_90_degrees_around_center(self):
      rows, cols = 3, 3
      image = np.array([
        [1,  2,  3],
        [4,  5,  6],
        [7,  8,  9],
      ], dtype=np.uint8)
      expected_approx = np.array([
        [7,  4,  1],
        [8,  5,  2],
        [9,  6,  3],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      rotated_image = mutator.applyRotate(image, math.pi / 2.0, 0.5, 0.5)

      np.testing.assert_array_almost_equal(expected_approx, rotated_image, decimal=0)

    def test_applyBarrelPincushionDistort_with_strength_0(self):
      rows, cols = 10, 10
      image = create_test_image(rows, cols, 128)

      mutator = ImageMutator(image)
      distorted_image = mutator.applyBarrelPincushionDistort(image, 0.0)

      np.testing.assert_array_equal(image, distorted_image)

    def test_applyBarrelPincushionDistort_with_positive_strength_barrel(self):
      rows, cols = 10, 10
      image = create_gradient_image(rows, cols)

      mutator = ImageMutator(image)
      distorted_image = mutator.applyBarrelPincushionDistort(image, 1.0)

      self.assertEqual(distorted_image[5, 5], image[5, 5])
      self.assertEqual(distorted_image[0, 0], 0)
      self.assertEqual(distorted_image[rows - 1, cols - 1], 0)

    def test_applyBarrelPincushionDistort_with_negative_strength_pincushion(self):
      rows, cols = 10, 10
      image = create_gradient_image(rows, cols)

      mutator = ImageMutator(image)
      distorted_image = mutator.applyBarrelPincushionDistort(image, -0.5)

      self.assertEqual(distorted_image[5, 5], image[5, 5])
      self.assertFalse(np.array_equal(image, distorted_image))

    def test_adjustBrightness_increases_brightness(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 100],
        [200, 240],
      ], dtype=np.uint8)
      expected = np.array([
        [30, 120],
        [220, 255],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      brightened_image = mutator.adjustBrightness(image, 20, 'positive')
      np.testing.assert_array_equal(expected, brightened_image)

    def test_adjustBrightness_decreases_brightness(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 100],
        [200, 240],
      ], dtype=np.uint8)
      expected = np.array([
        [0, 80],
        [180, 220],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      darkened_image = mutator.adjustBrightness(image, 20, 'negative')
      np.testing.assert_array_equal(expected, darkened_image)

    def test_adjustContrast_increases_contrast(self):
      rows, cols = 2, 2
      image = np.array([
        [50, 128],
        [128, 200],
      ], dtype=np.uint8)
      pivot = 128.0
      factor = 2.0

      expected = np.array([
        [0, 128],
        [128, 255],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      contrasted_image = mutator.adjustContrast(image, pivot, factor)
      np.testing.assert_array_equal(expected, contrasted_image)

    def test_adjustContrast_decreases_contrast(self):
      rows, cols = 2, 2
      image = np.array([
        [50, 128],
        [128, 200],
      ], dtype=np.uint8)
      pivot = 128.0
      factor = 0.5

      expected = np.array([
        [89, 128],
        [128, 164],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      decontrasted_image = mutator.adjustContrast(image, pivot, factor)
      np.testing.assert_array_equal(expected, decontrasted_image)

    def test_adjustGamma_with_gamma_1_0(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 100],
        [200, 240],
      ], dtype=np.uint8)
      original_image = image.copy()

      mutator = ImageMutator(image)
      gamma_image = mutator.adjustGamma(image, 1.0)
      np.testing.assert_array_equal(original_image, gamma_image)

    def test_adjustGamma_with_gamma_less_than_1_0_lighten(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 100],
        [200, 240],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      new_image = mutator.adjustGamma(image, 0.5)

      self.assertGreaterEqual(new_image[0, 0], 10)
      self.assertGreater(new_image[0, 1], 100)
      self.assertGreater(new_image[1, 0], 200)
      self.assertGreater(new_image[1, 1], 240)
      self.assertLessEqual(new_image[1, 1], 255)

    def test_adjustGamma_with_gamma_greater_than_1_0_darken(self):
      rows, cols = 2, 2
      image = np.array([
        [10, 100],
        [200, 240],
      ], dtype=np.uint8)

      mutator = ImageMutator(image)
      new_image = mutator.adjustGamma(image, 2.0)

      self.assertLessEqual(new_image[0, 0], 10)
      self.assertGreaterEqual(new_image[0, 0], 0)
      self.assertLess(new_image[0, 1], 100)
      self.assertLess(new_image[1, 0], 200)
      self.assertLess(new_image[1, 1], 240)

    def test_addNoiseGaussian_with_std_dev_0(self):
      rows, cols = 5, 5
      image = create_test_image(rows, cols, 100)

      mutator = ImageMutator(image)
      np.random.seed(0)
      noised_image = mutator.addNoiseGaussian(image, 0.0)

      np.testing.assert_array_equal(image, noised_image)

    def test_addNoiseGaussian_with_positive_std_dev(self):
      rows, cols = 5, 5
      image = create_test_image(rows, cols, 100)

      mutator = ImageMutator(image)
      np.random.seed(0)
      noised_image = mutator.addNoiseGaussian(image, 10.0)

      self.assertFalse(np.array_equal(image, noised_image))

      self.assertTrue(np.all(noised_image >= 0))
      self.assertTrue(np.all(noised_image <= 255))

    def test_addNoiseSaltAndPepper_with_probability_0(self):
      rows, cols = 5, 5
      image = create_test_image(rows, cols, 100)

      mutator = ImageMutator(image)
      np.random.seed(0)
      noised_image = mutator.addNoiseSaltAndPepper(image, 0.0)

      np.testing.assert_array_equal(image, noised_image)

    def test_addNoiseSaltAndPepper_with_probability_1(self):
      rows, cols = 5, 5
      image = create_test_image(rows, cols, 100)

      mutator = ImageMutator(image)
      np.random.seed(0)
      noised_image = mutator.addNoiseSaltAndPepper(image, 1.0)

      self.assertTrue(np.all((noised_image == 0) | (noised_image == 255)))

    def test_addNoiseSaltAndPepper_with_intermediate_probability(self):
      rows, cols = 10, 10
      image = create_test_image(rows, cols, 100)

      mutator = ImageMutator(image)
      np.random.seed(0)
      noised_image = mutator.addNoiseSaltAndPepper(image, 0.1)

      has_0_or_255 = np.any((noised_image == 0) | (noised_image == 255))
      has_100 = np.any(noised_image == 100)

      self.assertTrue(has_0_or_255)
      self.assertTrue(has_100)

    def test_applyConvolution_with_identity_kernel(self):
      rows, cols = 5, 5
      image = create_gradient_image(rows, cols)
      kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyConvolution(image, kernel, EdgeBehaviour.zero)

      np.testing.assert_array_almost_equal(image, convolved_image)

    def test_applyConvolution_with_simple_blur_kernel_zero_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel = np.array([
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
      ], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyConvolution(image, kernel, EdgeBehaviour.zero)

      expected_approx = np.array([
        [13, 23, 17],
        [30, 50, 36],
        [26, 43, 31],
      ], dtype=np.uint8)

      np.testing.assert_array_almost_equal(expected_approx, convolved_image, decimal=0)

    def test_applyConvolution_with_simple_blur_kernel_repeat_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel = np.array([
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
      ], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyConvolution(image, kernel, EdgeBehaviour.repeat)

      expected_approx = np.array([
        [23, 30, 36],
        [43, 50, 56],
        [63, 70, 76],
      ], dtype=np.uint8)

      np.testing.assert_array_almost_equal(expected_approx, convolved_image, decimal=0)

    def test_applyConvolution_with_simple_blur_kernel_max_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel = np.array([
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
      ], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyConvolution(image, kernel, EdgeBehaviour.max)

      expected_approx = np.array([
        [155, 108, 159],
        [115, 50, 121],
        [168, 128, 172],
      ], dtype=np.uint8)

      np.testing.assert_array_almost_equal(expected_approx, convolved_image, decimal=0)


    def test_applyRowConvolution_with_identity_kernel_zero_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel_r = np.array([0, 1, 0], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyRowConvolution(image, kernel_r, EdgeBehaviour.zero)

      np.testing.assert_array_almost_equal(image, convolved_image)

    def test_applyRowConvolution_with_simple_blur_kernel_repeat_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel_r = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyRowConvolution(image, kernel_r, EdgeBehaviour.repeat)

      expected_approx = np.array([
        [13, 20, 26],
        [43, 50, 56],
        [73, 80, 86],
      ], dtype=np.uint8)

      np.testing.assert_array_almost_equal(expected_approx, convolved_image, decimal=0)

    def test_applyColumnConvolution_with_identity_kernel_zero_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel_c = np.array([0, 1, 0], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyColumnConvolution(image, kernel_c, EdgeBehaviour.zero)

      np.testing.assert_array_almost_equal(image, convolved_image)

    def test_applyColumnConvolution_with_simple_blur_kernel_repeat_edge(self):
      rows, cols = 3, 3
      image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ], dtype=np.uint8)
      kernel_c = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

      mutator = ImageMutator(image)
      convolved_image = mutator.applyColumnConvolution(image, kernel_c, EdgeBehaviour.repeat)

      expected_approx = np.array([
        [20, 30, 40],
        [40, 50, 60],
        [60, 70, 80],
      ], dtype=np.uint8)

      np.testing.assert_array_almost_equal(expected_approx, convolved_image, decimal=0)

    def test_image_mutation_with_mnist(self):
      try:
        mnist_images_path = "./datasets/t10k-images.idx3-ubyte"
        mnist_labels_path = "./datasets/t10k-labels.idx1-ubyte"
        mnist_test = MnistIterator(mnist_images_path, mnist_labels_path)
        image, _ = next(iter(mnist_test))

        mutator = ImageMutator(image)

        random.seed(0)
        np.random.seed(0)

        mutator.applyTranslateX(image, 7, 'positive')
        mutator.applyTranslateX(image, 7, 'negative')
        mutator.applyTranslateY(image, 7, 'positive')
        mutator.applyTranslateY(image, 7, 'negative')
        mutator.applyScale(image, 2.0, 0.5, 0.5)
        mutator.applyScale(image, 0.5, 0, 0)
        mutator.applyScale(image, 0.5, 1, 1)
        mutator.applyScale(image, 0.5, 0.5, 0.5)
        mutator.applyRotate(image, math.pi/2.0, 0.5, 0.5)
        mutator.applyRotate(image, -math.pi/2.0, 0.5, 0.5)
        mutator.applyConvolution(image, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32), EdgeBehaviour.repeat)
        mutator.applyConvolution(image, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32), EdgeBehaviour.repeat)
        mutator.applyBarrelPincushionDistort(image, 1.0)
        mutator.applyBarrelPincushionDistort(image, -0.5)
        mutator.adjustBrightness(image, 20, 'positive')
        mutator.adjustContrast(image, 128, 0.5)
        mutator.adjustContrast(image, 128, 2.0)
        mutator.adjustGamma(image, 2.0)
        mutator.addNoiseGaussian(image, 64.0)
        mutator.addNoiseSaltAndPepper(image, 0.25)

        kernel_3x3_1d = np.array([1.0/4.0, 2.0/4.0, 1.0/4.0], dtype=np.float32)
        intermediate = mutator.applyColumnConvolution(image, kernel_3x3_1d, EdgeBehaviour.repeat)
        mutator.applyRowConvolution(intermediate, kernel_3x3_1d, EdgeBehaviour.repeat)

        kernel_3x3_1d_box = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
        intermediate = mutator.applyColumnConvolution(image, kernel_3x3_1d_box, EdgeBehaviour.repeat)
        mutator.applyRowConvolution(intermediate, kernel_3x3_1d_box, EdgeBehaviour.repeat)

        kernel_4x1_motion = np.array([1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0], dtype=np.float32)
        mutator.applyRowConvolution(image, kernel_4x1_motion, EdgeBehaviour.repeat)
        mutator.applyColumnConvolution(image, kernel_4x1_motion, EdgeBehaviour.repeat)

        kernel_4x4_lens = np.array([
          [0, 1.0/12.0, 1.0/12.0, 0],
          [1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0],
          [1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0],
          [0, 1.0/12.0, 1.0/12.0, 0]
        ], dtype=np.float32)
        mutator.applyConvolution(image, kernel_4x4_lens, EdgeBehaviour.repeat)


        self.assertGreater(len(mutator.images), 1)
        for img in mutator.images:
          self.assertEqual(img.shape, (28, 28))
          self.assertEqual(img.dtype, np.uint8)

      except FileNotFoundError:
        self.skipTest("MNIST dataset files not found. Skipping test_image_mutation_with_mnist.")
      except Exception as e:
        self.fail(f"An error occurred during MNIST mutation test: {e}")

  unittest.main(argv=['first-arg-is-ignored'], exit=False)

