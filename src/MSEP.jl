module MSEP

using DataFrames
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using NamedArrays
using QuadGK
using StatsFuns

# NormalGH is a struct and function name.  Not sure what the next line does.
export NormalGH
# same issue with
export Experiment
export MultiLevel

export AgnosticAGK

export maker
# and then there's the function associate with NormalGH instances
export condzy1
# other test code
export Experiment, ExperimentResult, compute, go

export LogisticSimpleEvaluator, Evaluator
export zSQdensity, wDensity, CTDensity, make_zAB_generator, make_zAS_generator, make_zCT_generator
export simulate, bigsim, msep, msepabs, bigbigsim, big3sim, rearrange, twoToOne

# includes at bottom

"""
Use Gauss-Hermite quadrature to evaluate a function.
Construct it by giving the number of points to evaluate and,
optionally, the mean and standard deviation of the distribution.

Thereafter you can treat the object as a function and call it on 
a function f that takes a single argument x.

The result is the f integrated over the normal distribution.
"""
struct NormalGH
    npoints
    μ
    σ
    w
    x
end

#constructor
# Using a "" comment here causes a warning about redefined documentation
function NormalGH(npoints; μ=0.0, σ=1.0)
    x, w = gausshermite(npoints)
    return NormalGH(npoints, μ, σ, w ./ sqrtπ, x)
end

"""evaluate the integral, treating `NormalGH` instance as a function
Integrate with respect to normal density.
"""
function (p::NormalGH)(f)
    return dot(p.w, f.(p.μ .+ sqrt2 * p.σ * p.x))
end

"""
Integrate using an adaptive Gauss-Kronrod method.
Assume the function passed in will be multiplied by
a standard normal density.

This is set up to use a specified order of integration,
which is almost completely undocumented.  It defaults to 7 and
the docs say it should be proportional to the desired precision.

Note that quadGK returns (estimate, error estimate)
"""
struct NormalAGK
    order::Int
    μ
    σ
    normal
end

# constructor
function NormalAGK(; μ=0.0, σ=1.0, order::Int=7)
    return NormalAGK(order, μ, σ, Normal(μ, σ))
end

"evaluate the integral"
function (nagk::NormalAGK)(f)
    value, err = quadgk(z -> f(z) * pdf(nagk.normal, z), -Inf, Inf,
        order=nagk.order, atol=sqrt(eps()))
    return value
end

"""
This evaluator assumes the density function is already built in
to the functions to be evaluated, or that you are just doing
a regular Adaptive Gauss Kronrod integral over the whole Real
line
"""
struct AgnosticAGK
    order
end

function (aagk::AgnosticAGK)(f; segbuf=nothing)
    value, err = quadgk(f, -Inf, Inf, order=aagk.order,
        atol=sqrt(eps()), segbuf=segbuf)
    return value
end

"""
conditional likelihood z|Y=1
for a single outcome with constant term k
The variance of the random effects is 1.
"""
function condzy1(z, k)
    return logistic(k + z) * pdf(Normal(), z)
end

"""
Returns a function which, when evaluated at z, gives
the product of weight (defined using zSQ with parameter λ) and
the standard normal density.

The hope is that considering them together will avoid overflow.
"""
function makezSQwd(λ)
    @assert λ < 0.5
    function (z)
        invsqrt2π * exp(-0.5 * (1.0 - 2.0 * λ) * z^2)
    end
end

"""
evaluate integrals of various functions with varying number of quadrature points
This is very special purpose code.
"""
struct Experiment
    "array of # of points to try"
    npoints
    "array of functions to integrate"
    funs
    "conditional likelihood(Y|z) within constant factor"
    condY

    # next terms computed rather than passed in
    "array of NormalGH for quadrature"
    quad_nodes

    "array of final functions to integrate over"
    fused_funs
end

"Experiment constructor"
function Experiment(npoints;
    funs=[z -> 1.0, identity, (z) -> exp(0.4 * z^2), (z) -> z * exp(0.4 * z^2)],
    condY=(z) -> logistic(z - 2))
    quad_nodes = [NormalGH(n) for n = npoints]
    fused = [z -> f(z) * condY(z) for f = funs]
    return Experiment(npoints, funs, condY, quad_nodes, fused)
end

function Experiment2(npoints;
    funs=[makezSQwd(0.4), z -> z * makezSQwd(0.4)(z)],#[z->1.0, identity, z->exp(0.4*z^2), z->z*exp(0.4*z^2)],
    condY=z -> logistic(z - 2))
    quad_nodes = [AgnosticAGK(n) for n = npoints]
    function maker(f)
        function inner(z)
            print("  @", z)
            a = f(z)
            print(" f(z)= ", a)
            b = condY(z)
            print(", Y|z= ", b)
            r = a * b
            println(" -> ", r)
            return r
        end
    end
    fused = [maker(f) for f = funs]
    return Experiment(npoints, funs, condY, quad_nodes, fused)
end
struct ExperimentResult
    "specification of Experiment"
    experiment::Experiment
    "rows are experiment.funs and columns experiment.npoints
    individual cells are computed quadrature values"
    result
end

function compute(experiment::Experiment)
    nfun = length(experiment.funs)
    nquad = length(experiment.npoints)
    # following seems to be only allowed call with type as first argument
    res = NamedArray(Real, nfun + 1, nquad)
    #setnames!(res, ["1", "z", "w", "wz", "zhat"], 1)
    setnames!(res, ["w", "wz", "zhat"], 1)
    setnames!(res, string.(experiment.npoints), 2)
    setdimnames!(res, "f", 1)
    setdimnames!(res, "npoints", 2)
    for i1 = 1:nquad
        println("order ", experiment.npoints[i1])
        for i0 = 1:nfun
            println("  Function # ", i0)
            res[i0, i1] = experiment.quad_nodes[i1](experiment.fused_funs[i0])
        end
        res[nfun+1, i1] = res["wz", i1] / res["w", i1]
    end
    return ExperimentResult(experiment, res)
end

function test()
    expt = Experiment2([1, 2, 3]) # [1, 3, 5, 7, 8, 9, 10, 15, 20])
    r = compute(expt)
    display(r.result)
    return r
end

# evaluator needs some of the classes defined above
include("maker.jl")
include("evaluator.jl")
include("post.jl")
include("bigbigsim.jl")

end
