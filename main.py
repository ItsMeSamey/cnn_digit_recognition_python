from cnn import CnnTester
from functions_activate import NormalizeSquared, PReLU
from functions_loss import MeanSquaredError
from layers import ConvolveTester, DenseTester, FlattenTester, ParallelTester, SequentialTester
from read_mnist import mnist_test_iter

tester = CnnTester((28, 28), MeanSquaredError(), SequentialTester([
  ParallelTester([
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
  ]),

  FlattenTester(),

  DenseTester(14*14, PReLU(0.125)),
  DenseTester(14*14, PReLU(0.125)),
  DenseTester(7*7, PReLU(0.125)),
  DenseTester(7*7, PReLU(0.125)),
  DenseTester(10, NormalizeSquared()),
]), 'db8b2111e6fb98767e9335b691a23d4a')

print("Loading model...")
tester.load()
tester.hash = 'test'
print("Saving model...")
tester.save()

print("Loading model...")
tester.load()

print("Testing...")
accuracy = tester.test(mnist_test_iter, True)

print("Accuracy: %.2f%%" % (accuracy*100))

