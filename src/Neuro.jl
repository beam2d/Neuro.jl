# Framework for learning and applying neural networks.
module Neuro

# Layers of neural network
include("layer/Layer.jl")

# Architectures of networks
include("network/Network.jl")

# Gradient-based learners
include("learner/Learner.jl")

# Utilities for reading datasets
include("dataset/Dataset.jl")

end
