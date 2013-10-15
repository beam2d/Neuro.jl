export Layers

# Sorts of layers
module Layers
using ..Neuro

# Learnable layers
include("dense_layer.jl")

# Immutable layers
include("immutable_layer.jl")

include("relu_layer.jl")
include("softmax_layer.jl")

end
