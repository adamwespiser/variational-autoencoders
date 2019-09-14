
include("Dataset.jl")
import .Dataset:
  get_MINST,
  TrainTest,
  split_X_by_batches

include("Model.jl")
import .Model:
  encoder,
  decoder,
  split_encoder_result,
  random_sample_decode,
  create_vae

include("Utils.jl")
import .Utils:
  gen_images

using Test
using Flux
using Flux.Tracker: TrackedReal
using Printf

function run_batches()
  n_traning_examples = 6400
  batch_size = 64
  n_latent = 10
  dataset = get_MINST(n_traning_examples)
  ps, loss_fn, f, g = create_vae(n_latent, batch_size)
  X_batches = split_X_by_batches(dataset.train_x, batch_size)
  opt = ADAM()
  if !isdir("output")
    mkdir("output")
  end
  for epoch_idx=1:5
    Flux.train!(loss_fn, ps, X_batches, opt)
    loss_test_set = loss_fn(dataset.test_x[:,:,:,1:batch_size])
    @info(@sprintf("[%d] test loss : %.4f",epoch_idx, loss_test_set))

    epoch_img = joinpath(pwd(), "output" , @sprintf("img_%d.png", epoch_idx))
    gen_images(epoch_img, g, f, dataset.test_x)
  end
end

run_batches()
