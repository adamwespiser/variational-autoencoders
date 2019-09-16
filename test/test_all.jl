include("../src/Dataset.jl")
import .Dataset:
  get_MINST,
  TrainTest,
  split_X_by_batches

include("../src/Model.jl")
import .Model:
  encoder,
  decoder,
  split_encoder_result,
  create_vae,
  model_sample

include("../src/Utils.jl")
import .Utils:
  gen_images


using Test
using Flux
using Flux.Tracker: TrackedReal
using Printf

temp_dir = tempdir()

@testset "Image Utilities" begin
  n_sample = 10
  n_latent = 10
  dataset = get_MINST(n_sample)
  ps, loss_fn, f, g = create_vae(n_latent, n_sample)

  # zero initialized μ̂, logσ̂
  outfile_zero_latent = joinpath(temp_dir, "zero_latent.png")
  @test typeof(model_sample(f)) == BitArray{4}
  gen_images(outfile_zero_latent, f)
  @test isfile(outfile_zero_latent)
  @sprintf("saved file to %s", outfile_zero_latent)

  # f(g(x)) test
  outfile_X_latent = joinpath(temp_dir, "X_latent.png")
  gen_images(outfile_X_latent, g, f, dataset.test_x)
  @test isfile(outfile_X_latent)
  @sprintf("saved file to %s", outfile_X_latent)
end


@testset "ADAM optimization can run" begin
  n_sample = 1
  n_latent = 10
  dataset = get_MINST(n_sample)
  X = dataset.train_x
  ps, loss_fn, f, g = create_vae(n_latent, n_sample)
  opt = ADAM()

  @test typeof(loss_fn(X)) == TrackedReal{Float64}
  X = float.(X .> 0.5)
  Flux.train!(loss_fn, ps, zip([X]), opt)
  @test true == true
end


@testset "Convolution and transpose is isomorphic" begin
  dataset = get_MINST()

  n_sample = 100
  n_latent = 10
  X = dataset.train_x[:,:,:,1:n_sample]
  enc_model = encoder(n_latent)
  X_transformed = enc_model(reshape(X, 28, 28, 1, n_sample))

  x_mean, x_std = split_encoder_result(X_transformed, n_latent)

  dec_model = decoder(n_latent)
  Xp = dec_model(x_mean)
  @test size(Xp) == size(X)
end


@testset "MINST dataset: size/shape okay" begin
  ds = get_MINST()
  img_shape = (28, 28, 1)
  n_train = 60000
  n_test = 10000
  @test size(ds.train_x) == (img_shape..., n_train)
  @test size(ds.train_y) == (n_train, )
  @test size(ds.test_x) ==  (img_shape..., n_test)
  @test size(ds.test_y) == (n_test, )
end
