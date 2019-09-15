module Utils

include("Model.jl")
import .Model:
  model_sample

using Images:
  Gray,
  save

function img(x)
  Gray.(rotate(reshape(x, 28, 28)))
end

function gen_images(outfile, f)
  sample = hcat(img.([model_sample(f) for i = 1:10])...)
  save(outfile, sample)
end

function gen_images(outfile, g, f, X)
  sample = hcat(img.([model_sample(g, f, X[:,:,:,i]) for i = 1:size(X,4)])...)
  save(outfile, sample)
end

function rotate(ximg::Array{T, 2}) where {T}
  i_idx = size(ximg, 1)
  j_idx = size(ximg, 2)
  xnew = zeros(T, i_idx, j_idx)
  for i in range(1, stop = i_idx)
    for j in  range(1, stop = j_idx)
      xnew[i,j] = ximg[j,i]
    end
  end
  xnew
end
