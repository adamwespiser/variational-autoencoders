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
  n_sample_train = 60000
  n_sample_test = 10000
  x_train_whcn = reshape(float.(train_x[:,:,1:n_sample_train]), 28, 28, 1, n_sample_train)
  x_test_whcn = reshape(float.(test_x[:,:,1:n_sample_test]), 28, 28, 1, n_sample_test)

  return TrainTest(
    float.(x_train_whcn .> 0.5),
    train_y[1:n_sample_train],
    float.(x_test_whcn .> 0.5),
    test_y[1:n_sample_test])
end


function get_MINST(n_sample :: Int)
  @assert n_sample < 10000
  test_x, test_y = MNIST.testdata()
  train_x, train_y = MNIST.traindata()

  x_train_whcn = reshape(float.(train_x[:,:,1:n_sample]), 28, 28, 1, n_sample)
  x_test_whcn = reshape(float.(test_x[:,:,1:n_sample]), 28, 28, 1, n_sample)

  return TrainTest(
    float.(x_train_whcn .> 0.5),
    train_y[1:n_sample],
    float.(x_test_whcn .> 0.5),
    test_y[1:n_sample])
end


function split_X_by_batches(
  X :: Array{T, 4},
  batch_size :: Int64
  ) where {T}
  # X is in (width, height, channel, samples)
  N_total = size(X, 4)
  batch_idx = collect(Base.Iterators.partition(1:N_total, batch_size))
  return zip([X[:,:,:,idx] for idx in batch_idx])
end

end
