
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

using Test
using Flux
using Flux.Tracker: TrackedReal
using Printf

function run_batches()
  n_traning_examples = 640
  batch_size = 64
  n_latent = 10
  dataset = get_MINST(n_traning_examples)
  ps, loss_fn, f, g = create_vae(n_latent, batch_size)
  X_batches = split_X_by_batches(dataset.train_x, batch_size)
  opt = ADAM()
  for epoch_idx=1:100
    Flux.train!(loss_fn, ps, X_batches, opt)
    loss_test_set = loss_fn(dataset.test_x[:,:,:,1:batch_size])
    @info(@sprintf("[%d] test loss : %.4f",epoch_idx, loss_test_set))
  end
end

run_batches()

function test_adam_step()
  n_sample = 1
  n_latent = 10
  dataset = get_MINST(n_sample)
  X = dataset.train_x
  println(typeof(X))
  println(size(X))
  ps, loss_fn, f, g = create_vae(n_latent, n_sample)
  opt = ADAM()
  @test typeof(loss_fn(X)) == TrackedReal{Float64}

  X = float.(X .> 0.5)
  Flux.train!(loss_fn, ps, zip([X]), opt)

  @test true == true
end

test_adam_step()

function test_conv_deconv()
  dataset = get_MINST()

  n_sample = 100
  n_latent = 10
  X = convert(Array{Float32,3},dataset.train_x[:,:,1:n_sample])
  enc_model = encoder(n_latent)
  X_transformed = enc_model(reshape(X, 28, 28, 1, n_sample))

  x_mean, x_std = split_encoder_result(X_transformed, n_latent)

  dec_model = decoder(n_latent)
  Xp = dec_model(x_mean)
  @test size(Xp) == size(X)
end

test_conv_deconv()

