
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
  gen_images,
  img

using Test
using Flux
using Flux.Tracker: TrackedReal
using Printf
using Images

function run_batches()
  n_traning_examples = 60000
  batch_size = 128
  n_latent = 10
  n_test_viz = 10
  max_epoch_idx = 20
  output_dir = joinpath(pwd(), "output")
  dataset = get_MINST()
  x_view = dataset.test_x[:,:,:,dataset.test_y .==  5]
  ps, loss_fn, f, g = create_vae(n_latent, batch_size)

  X_batches = split_X_by_batches(dataset.train_x, batch_size)
  opt = ADAM()
  # Make output director
  if !isdir(output_dir)
    mkdir(output_dir)
  end
  # save test images (ignored by git)

  x_imgs_test = hcat(img.([x_view[:,:,:,i] for i = 1:n_test_viz])...)
  save(joinpath(output_dir, "test.png"), x_imgs_test)

  for epoch_idx = range(1, stop = max_epoch_idx)
    for (data_idx, dataset_batch) in enumerate(X_batches)
      start = time()
      Flux.train!(loss_fn, ps, [dataset_batch], opt)
      loss_test_set = loss_fn(dataset.test_x[:, :, :, 1:batch_size])
      @info @sprintf("[%d][%d] test loss : %.4f", epoch_idx, data_idx, loss_test_set)
      @info @sprintf("[%d][%d] elapsed %d(s)", epoch_idx, data_idx, time() - start)
      if data_idx % 5 == 1
        epoch_img = joinpath(output_dir, @sprintf("img_epoch=%d_batch=%d.png", epoch_idx, data_idx))
        gen_images(epoch_img, g, f, x_view[:, :, :, 1:n_test_viz])
      end
    end
  end
end

run_batches()
