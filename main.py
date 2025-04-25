from cnn import CnnTester
from functions_activate import NormalizeSquared, PReLU
from functions_loss import MeanSquaredError
from layers import ConvolveTester, DenseTester, FlattenTester, LRFnWrappedTester, ParallelTester, SequentialTester
from read_mnist import mnist_test_iter

tester = CnnTester((28, 28), MeanSquaredError(), SequentialTester([
  LRFnWrappedTester(ParallelTester([
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
    ConvolveTester(8, 8, 4, 4, PReLU(0.125)),
  ]), lambda rate, layer: rate / len(layer.layers)),

  FlattenTester(),

  DenseTester(14*14, PReLU(0.125)),
  DenseTester(14*14, PReLU(0.125)),
  DenseTester(7*7, PReLU(0.125)),
  DenseTester(7*7, PReLU(0.125)),
  DenseTester(10, NormalizeSquared()),
]), 'db8b2111e6fb98767e9335b691a23d4a')

# trainer = tester.to_trainer()
# trainer.train(mnist_test_iter, 1, 32, True)

# print("Loading model...")
# tester.load()
# tester.hash = 'test'
# print("Saving model...")
# tester.save()

print("Loading model...")
tester.load()

print("Testing...")
accuracy = tester.test(mnist_test_iter, True)

print("Accuracy: %.2f%%" % (accuracy*100))

