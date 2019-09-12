module Dataset
export TrainTest, get_MINST

using MLDatasets


struct TrainTest
  train_x
  train_y
  test_x
  test_y
end

function get_MINST()
  test_x, test_y = MNIST.testdata()
  train_x, train_y = MNIST.traindata()
  return TrainTest(train_x, train_y, test_x, test_y)
end

function get_MINST(n_sample :: Int)
  test_x, test_y = MNIST.testdata()
  train_x, train_y = MNIST.traindata()

  x_train_whcn = reshape(float.(train_x[:,:,1:n_sample]), 28, 28, 1, n_sample)
  x_test_whcn = reshape(float.(test_x[:,:,1:n_sample]), 28, 28, 1, n_sample)

  return TrainTest(
    x_train_whcn,
    train_y[1:n_sample],
    x_test_whcn,
    test_y[1:n_sample])
end


end
