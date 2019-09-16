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

function repr_digit(
  X,
  y,
  digit :: Int64
  )
  # digit in 1:10

  digit_idx = findall(y .== digit)
  Xdigit = X[:,:,:, digit_idx]
  n_examples = size(Xdigit,4)
  mat = reshape(Xdigit, n_examples, 28*28)
  dist_mat = zeros(n_examples, n_examples)
  distance(x,y) = sum((x .- y).^ 2)
  for i in range(1, stop = n_examples)
    for j in range(1, stop = n_examples)
      dist_mat[i,j] = distance(mat[i,:], mat[j,:])
    end
  end

  avg_dist = reshape(sum(dist_mat, dims = 1), n_examples)
  digit_idx[findall(avg_dist .== minimum(avg_dist))[1]]
end

function average_digits()
  test_x, test_y = MNIST.testdata()
  train_x, train_y = MNIST.traindata()

  x_train_whcn = reshape(float.(train_x[:,:,:]), 28, 28, 1, :)
  x_test_whcn = reshape(float.(test_x[:,:,:]), 28, 28, 1, :)

  X = x_test_whcn
  y = test_y
  [repr_digit(X, y, i) for i in 1:9]
end

function test_viz_idx()
  return viz_idx = [2380, 1304, 9008, 766, 711, 2472, 7761, 7756, 7953]
end

end
