# Neural Network Project

This project is a basic implementation of a convolutional neural network from scratch in Python using NumPy only, final model had accuracy of 99.24% on MNIST test set.
This project also includes an interactive web demo.

There is a [Zig version](github.com/ItsMeSamey/cnn_digit_recognition_python) with same functionality (and same model format) but with higher performance and more functionality.

> ![NOTE]
> You should consider using `numba` as python3 runtime for a significant speedup.

## Usage

1. Ensure you have Python installed.
2. Clone the repository or download the project files using
```bash
git clone --depth=1 https://github.com/ItsMeSamey/cnn_digit_recognition_python.git
```
3. Install necessary libraries (e.g., NumPy):
```bash
pip install numpy
```

### Tresting accuracy on MNIST dataset
Run `main.py` to test the network on the MNIST dataset.
```bash
python3 main.py
```

### Training the network
delete the `model_` files (both `.json` and `.cnn`) and then run the `main.py` script, the training will start automatically.
```bash
rm model_*
python3 main.py
```

### Demo Application
1. Run the server using:
```bash
python3 server.py
```
2. Open the `demo.html` file in your web browser to launch interactive demo.

## File Structure

- `cnn.py`: Contains the main CNN class, CnnTrainer, and CnnTester for building, training, and evaluating the network.
- `layers.py`: Defines the different layer types (ConvolveTester/Trainer, DenseTester/Trainer, etc.) and their forward/backward logic.
- `functions_activate.py`: Implements the various activation functions.
- `functions_loss.py`: Implements the different loss functions.
- `read_mnist.py`: Handles reading the MNIST dataset files and provides data iterators.
- `image_operations.py`: Contains functions for image augmentation.

- `main.py`: Contains the main script for training and testing.

- `server.py`: A web server to handle requests from the demo application.
- `demo.html`: The frontend HTML and JavaScript for the web demo.


## License
This project is licensed under the `GNU AFFERO GENERAL PUBLIC LICENSE Version 3` - see the [LICENSE](LICENSE) file for details.

