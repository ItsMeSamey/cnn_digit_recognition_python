from cnn import CnnTester, print_test_results
from functions_activate import NormalizeSquared, PReLU
from functions_loss import MeanSquaredError
from layers import ConvolveTester, DenseTester, FlattenTester, LRFnWrappedTester, ParallelTester, SequentialTester
from read_mnist import EqualizedIterator, MutatorWrapped, RandomMnistIterator, mnist_test_iter, mnist_train_iter

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
    for i in range(9):
      random_iter = RandomMnistIterator(mnist_train_iter, mnist_train_iter.count*4*(i+1))
      mutator = MutatorWrapped(random_iter)
      equalized = EqualizedIterator(mutator)
      trainer.train(equalized, 1 / pow(2, i), mnist_train_iter.count // (10*(i+1)*(i+1)), True)

    for i in range(8):
      random_iter = RandomMnistIterator(mnist_train_iter, mnist_train_iter.count*10*(i+1))
      equalized = EqualizedIterator(random_iter)
      trainer.train(equalized, 1 / pow(2, i), mnist_train_iter.count // (10*(i+1)*(i+1)), True)

    for i in range(9):
      random_iter = RandomMnistIterator(mnist_train_iter, mnist_train_iter.count*2)
      mutator = MutatorWrapped(random_iter)
      mutator = MutatorWrapped(mutator)
      equalized = EqualizedIterator(mutator)
      trainer.train(equalized, 0.5, mnist_train_iter.count // (10*(i+1)*(i+1)), True)

    for i in range(8):
      random_iter = RandomMnistIterator(mnist_train_iter, mnist_train_iter.count*10*(i+1))
      equalized = EqualizedIterator(random_iter)
      trainer.train(equalized, 1, True)

    tester.save()
else:
  tester.load()

if __name__ == '__main__':
  # tester.save()
  # tester.load()
  print("Testing...")
  results = tester.test(mnist_test_iter, True)
  print_test_results(results)

