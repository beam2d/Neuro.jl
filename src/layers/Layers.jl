export Layers

# Sorts of layers
module Layers
using ..Neuro

include("dense_layer.jl")
include("relu_layer.jl")
include("softmax_layer.jl")

end
