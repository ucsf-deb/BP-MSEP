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
export MultiLevel

export AgnosticAGK

export maker
# and then there's the function associate with NormalGH instances
export condzy1


export LogisticABEvaluator, LogisticBPEvaluator, LogisticSimpleEvaluator, LogisticCutoffEvaluator, Evaluator, CutoffAGK
export name, description, name_with_suffix, zhat, zsimp
export zSQdensity, wDensity, make_zAB_generator, make_zAS_generator, make_zCT_generator
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
    δ = sqrt(eps())
    value, err = quadgk(f, -Inf, Inf, order=aagk.order,
        atol=δ, segbuf=segbuf)
    if err > 2*max(δ, δ*value)
        error("Unable to integrate accurately")
    end
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


# evaluator needs some of the classes defined above
include("maker.jl")
include("evaluator.jl")
include("logistic_simple_evaluator.jl")
include("logistic_cutoff_evaluator.jl")
include("logistic_AB_evaluator.jl")
include("logistic_BP_evaluator.jl")
include("simulate.jl")
include("post.jl")
include("bigbigsim.jl")

end
