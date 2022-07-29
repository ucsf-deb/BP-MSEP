module MSEP

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using NamedArrays
using StatsFuns

# NormalGH is a struct and function name.  Not sure what the next line does.
export NormalGH
# same issue with
export Experiment

# and then there's the function associate with NormalGH instances
export condzy1
# other test code
export Experiment, ExperimentResult, compute, go

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

"constructor"
function NormalGH(npoints; μ=0.0, σ=1.0)
    x, w = gausshermite(npoints)
    return NormalGH(npoints, μ, σ, w ./ sqrtπ, x)
end

"""evaluate the integral, treating `NormalGH` instance as a function
Integrate with respect to normal density.
"""
function (p::NormalGH)(f)
    return dot(p.w, f.(p.μ.+sqrt2*p.σ*p.x))  
end

"""
conditional likelihood z|Y=1
for a single outcome with constant term k
The variance of the random effects is 1.
"""
function condzy1(z, k)
    return logistic(k+z)*pdf(Normal(), z)
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
        funs=[identity, (z)->exp(0.4*z^2), (z)->z*exp(0.4*z^2)],
        condY=(z)->logistic(z-2))
    quad_nodes = [NormalGH(n) for n=npoints]
    fused = [z -> f(z)*condY(z) for f = funs]
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
    res = NamedArray(Real, nfun, nquad)
    setnames!(res, ["z", "w", "wz"], 1)
    colnames = string.(experiment.npoints)
    setnames!(res, string.(experiment.npoints), 2)
    setdimnames!(res, "f", 1)
    setdimnames!(res, "npoints", 2)
    for i0 = 1:nfun
        for i1 = 1:nquad
            res[i0, i1] = experiment.quad_nodes[i1](experiment.fused_funs[i0])
        end
    end
    return ExperimentResult(experiment, res)
end

function test()
    expt = Experiment([1, 3, 5, 7, 8, 9, 10, 15, 20])
    r = compute(expt)
    display(r.result)
    return r
end
test()
end
