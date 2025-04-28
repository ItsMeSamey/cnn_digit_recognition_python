from cnn import CnnTester
from functions_activate import NormalizeSquared, PReLU
from functions_loss import MeanSquaredError
from layers import ConvolveTester, DenseTester, FlattenTester, LRFnWrappedTester, ParallelTester, SequentialTester
from read_mnist import mnist_test_iter

nnlayer = lambda _: SequentialTester([
  ConvolveTester(4, 4, 1, 1, PReLU(0.125)),
  ConvolveTester(2, 2, 2, 2, PReLU(0.125)),
  FlattenTester(),
  DenseTester(7*7, PReLU(0.125)),
  DenseTester(10, PReLU(0.125)),
])

tester = CnnTester((28, 28), MeanSquaredError(), SequentialTester([
  LRFnWrappedTester(ParallelTester([
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
    nnlayer(0),
  ]), lambda rate, layer: rate / len(layer.layers)),

  DenseTester(100, PReLU(0.125)),
  DenseTester(100, PReLU(0.125)),
  DenseTester(50, PReLU(0.125)),
  DenseTester(50, PReLU(0.125)),
  DenseTester(10, NormalizeSquared()),
]), '284926022fc8da7df328b6a60eaf2b31')

exists = tester.exists()
if not exists:
  print("Model does not exist, please train the model first.")
  if __name__ != '__main__':
    exit(1)
  else:
    trainer = tester.to_trainer()
    trainer.train(mnist_test_iter, 1, 32, True)
    tester.save()
else:
  tester.load()

if __name__ == '__main__':
  # tester.save()
  # tester.load()
  print("Testing...")
  accuracy = tester.test(mnist_test_iter, True)
  print("Accuracy: %.2f%%" % (accuracy*100))

