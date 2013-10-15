using Base.Test
using Neuro.Layers

# Test on a sample
let
    l = SoftmaxLayer{Float64}()
    x = [1., 0., -1.]
    y = fprop(l, x)
    @test_approx_eq([e, 1, 1/e] / (e + 1 + 1/e), y)

    groundtruth = [0., 1., 0.]
    @test l == grad(l, x, groundtruth)
    @test_approx_eq(y - groundtruth, bprop(l, x, y, groundtruth))

    @test size(weight(l)) == (0,)
end

# Test on a minibatch
let
    l = SoftmaxLayer{Float64}()
    x1 = [1., 0., -1.]
    x2 = [2., 1., -1.]
    y = fprop(l, [x1 x2])

    y1_expect = [e, 1, 1/e] / (e + 1 + 1/e)
    y2_expect = [e * e, e, 1/e] / (e * e + e + 1/e)
    @test_approx_eq([y1_expect y2_expect], y)

    groundtruth1 = [0., 1., 0.]
    groundtruth2 = [1., 0., 0.]
    groundtruth = [groundtruth1 groundtruth2]
    @test l == grad(l, [x1 x2], groundtruth)
    @test_approx_eq(y - groundtruth, bprop(l, [x1 x2], y, groundtruth))
end
