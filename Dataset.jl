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

end
