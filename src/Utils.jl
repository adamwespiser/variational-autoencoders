module Utils

include("Model.jl")
import .Model:
  model_sample

using Printf
using Images:
  Gray,
  save

using BSON:
  @save,
  @load


function img(x)
  Gray.(rotate(reshape(x, 28, 28)))
end

function gen_images(outfile, f)
  sample = hcat(img.([model_sample(f) for i = 1:10])...)
  save(outfile, sample)
end

function gen_images(
  outfile :: String,
  g,
  f,
  X :: T
) where {T <: AbstractArray}
  sample = hcat(img.([model_sample(g, f, X[:,:,:,i]).data for i = 1:size(X,4)])...)
  safe_img_save(outfile, sample)
end

function safe_img_save(x, outfile)
  try
    save(x, outfile)
  catch e
    @error @sprintf("failed write: %s", outfile) e
  end
end


function rotate(ximg :: T) where {T <: AbstractArray}
  xcopy = zeros(size(ximg)...)
  for i in range(1, stop = size(ximg, 1))
    for j in  range(1, stop = size(ximg, 2))
      xcopy[i,j] = ximg[j,i]
    end
  end
  xcopy
end

function save_model(f, g, outfile)
  model_pair = (f,g)
  try
    @save outfile model_pair
  catch e
    @error @sprintf("failed write model to: %s", outfile) e
  end
end

function load_model(infile)
  @load infile model_pair
  f,g = model_pair
  return f, g
end

end
