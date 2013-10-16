module ChainNetworkTest

using Base.Test
using Neuro
using Neuro.Layers
using Neuro.Networks

# Mock layer that multiplies a scalar.
immutable MultiplierLayer <: Layer{Float64}
    multiplier::Vector{Float64}

    MultiplierLayer(m::Float64) = new([m])
end

Layers.fprop(l::MultiplierLayer, in::Array{Float64}) = l.multiplier[1] * in
Layers.grad(l::MultiplierLayer, in::Array{Float64}, d_out::Array{Float64}) =
    MultiplierLayer(sum(in .* d_out))
Layers.bprop(l::MultiplierLayer, in::Array{Float64}, out::Array{Float64}, d_out::Array{Float64}) =
    l.multiplier[1] * d_out
Layers.weight(l::MultiplierLayer) = Array{Float64}[l.multiplier]

# Mock layer that adds a constant.
type ConstantAdderLayer <: ImmutableLayer{Float64}
    constant::Float64
end

Layers.fprop(l::ConstantAdderLayer, in::Array{Float64}) = in .+ l.constant
Layers.bprop(l::ConstantAdderLayer, in::Array{Float64}, out::Array{Float64}, d_out::Array{Float64}) = d_out

# Usual case
let
    chain = Layer{Float64}[
        MultiplierLayer(2.),
        ConstantAdderLayer(1.),
        MultiplierLayer(3.)
    ]
    x = [1., 2.]
    @test [9., 15.] == Networks.fprop(chain, x)

    activations = calc_activations(chain, x)
    @test 4 == length(activations)
    @test x == activations[1]
    @test [2., 4.] == activations[2]
    @test [3., 5.] == activations[3]
    @test [9., 15.] == activations[4]

    grads = grad(chain, activations, [1., -1.])
    @test 3 == length(grads)
    @test [-3.] == grads[1].multiplier
    @test 1. == grads[2].constant
    @test [-2.] == grads[3].multiplier

    weights = weight(chain)
    @test 2 == length(weights)
    @test chain[1].multiplier == weights[1]
    @test chain[3].multiplier == weights[2]
end

end
