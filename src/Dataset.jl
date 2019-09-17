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

# take a smaller sample of
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
  """
  split_X_by_batches(X, batch_size)
  given X is an Array in (width, height, channel, samples)
  we will further slice X into batches and return a result
  directly passable to Flux.Train!
  """
  N_total = size(X, 4)
  batch_idx = collect(Base.Iterators.partition(1:N_total, batch_size))
  return zip([X[:,:,:,idx] for idx in batch_idx])
end

function repr_digit(
  X :: U,
  y :: Array{T, 1},
  digit :: T
) where {T, U <: AbstractArray}
  @assert digit in unique(y) #ugh, how can this be a total fn?

  digit_idx = findall(y .== digit)
  Xdigit = X[:,:,:, digit_idx]
  n_examples = size(Xdigit,4)
  mat = reshape(Xdigit, n_examples, 28*28)
  dist_mat = zeros(n_examples, n_examples)
  distance(x,y) = sum((x .- y) .^ 2)
  for i in range(1, stop = n_examples)
    for j in range(1, stop = n_examples)
      dist_mat[i,j] = distance(mat[i,:], mat[j,:])
    end
  end

  avg_dist = reshape(sum(dist_mat, dims = 1), n_examples)
  digit_idx[findall(avg_dist .== minimum(avg_dist))[1]]
end

function min_distance_digits()
  """
  min_distance_digits
  Returns the index of a sample set of digits 0-9 in the test set
  with the condition that the example for each digit is the digit
  with the minimum average distance (L2) from that digit and all
  other digits of the same numeral.
  """
  test_x, test_y = MNIST.testdata()
  x_test_whcn = reshape(float.(test_x[:,:,:]), 28, 28, 1, :)

  [repr_digit(x_test_whcn, test_y, i) for i in sort(unique(test_y))]
end

function test_viz_idx()
  """
  test_viz_idx()
  returns the results of min_distance_digits for 0-9
  when applied to MINST test set
  """
  return [
    9426, # 0
    2380,
    1304,
    9008,
    766,
    711,
    2472,
    7761,
    7756,
    7953 # 9
  ]
end

end
