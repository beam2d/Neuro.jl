using Base.Test
using Neuro.Layer

# Test on a sample
let
    l = ReluLayer{Float64}()
    x = [1., 0., -1.]
    y = fprop(l, x)
    @test_approx_eq([1., 0., 0.], y)

    d = [2., 1., 2.]
    @test length(grad(l, x, d)) == 0
    @test_approx_eq([2., 0., 0.], bprop(l, x, y, d))

    @test size(weight(l)) == (0,)
end

# Test on a minibatch
let
    l = ReluLayer{Float64}()
    x1 = [1., 0., -1.]
    x2 = [-2., 1., 2.]
    y = fprop(l, [x1 x2])
    @test_approx_eq([1. 0.; 0. 1.; 0. 2.], y)

    d1 = [2., 1., 2.]
    d2 = [0., -1., 1.]
    @test length(grad(l, [x1 x2], [d1 d2])) == 0
    @test_approx_eq([2. 0.; 0. -1.; 0. 1.], bprop(l, [x1 x2], y, [d1 d2]))
end
