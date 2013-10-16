export Layer

# Sorts of layers
module Layer
using ..Neuro

include("abstract_layer.jl")

# Learnable layers
include("dense_layer.jl")

# Immutable layers
include("immutable_layer.jl")

include("relu_layer.jl")
include("softmax_layer.jl")

end
